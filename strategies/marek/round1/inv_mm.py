"""inv_mm: ladder_mm (strategies/peter/ladder_mm.py) plus inventory-aware switches, for the round1 inventory
study (research/round1/inventory.md). With every new switch at its default this is ladder_mm exactly.

New switches (on top of ladder_mm's skew_lots / improve_reduce / flat_at):
    lim_add     max |position| the side that adds to the current position may reach (100 = ladder_mm)
    lim_flip    max position on the other side of flat the reducing side may reach (100 = ladder_mm)
    size_add    >0: target lots on the adding side while |pos| > 0 (instead of size_top)
    add_back    >0: once |pos| >= add_back_at, quote the adding side this many ticks behind the touch
    end_us / unwind_s   after end_us - unwind_s: stop adding, only rest the reducing side up to flat
    unwind_ioc_s        >0: in the last unwind_ioc_s seconds IOC the position flat at the touch (once per s)
    adapt_w_s   >0: regime switch. Track our own mark-to-mid PnL (cash + pos * mid) sampled every second; while
                its change over the last adapt_w_s seconds is below adapt_thr, apply skew_lots = adapt_skew and
                lim_add = adapt_lim (inventory control only when the market is beating us)

ladder_mm docstring follows.

ladder_mm: passive market maker that rests large orders at and behind the touch and keeps queue priority.

Idea (from the sample_v2 first 120 s): the price barely moves and all trading comes from large,
uninformed IOC orders, so the edge is being filled by them at the touch. Queue position matters
more than price: an order whose price is still wanted is never re-sent.

Per side, the wanted prices are the touch (`touch` = best market price, our own orders excluded)
and `levels - 1` ticks behind it; `size_top` lots go to the first, `size_deep` to each of the rest.
A guard keeps the worst case |position + every open order on a side| within `cap`, so the
+-100 penalty can't trigger with cap <= 100.

Tried and kept behind flags (all hurt on sample_v2, see the git history):
    skew_lots        shift the adding side's ladder away by pos/skew_lots ticks
    improve_reduce   step inside a 2-tick spread on the side that reduces inventory
    flat_at/flat_to  IOC the position back to flat_to once |pos| >= flat_at
"""

from __future__ import annotations

from simulator.models import (
    OrderBookSnapshot,
    OrderRequest,
    OrderStatus,
    OrderType,
    PrivateOrderEvent,
    Side,
    Trade,
)
from simulator.strategy import BaseStrategy

PARAMS = {
    'size_top': 200,  # lots at the touch (clipped by the guard; 200 = cap + a full opposite position)
    'size_deep': 0,  # lots at each level behind the touch
    'levels': 1,  # price levels per side, touch included
    'cap': 100,  # worst-case |position + open orders on a side|
    'max_spread': 8,  # ignore books wider than this (a side backed by stale far orders)
    'tight_offset': 0,  # 1: with a 1-tick spread, start the ladder one tick behind the touch
    'topup': 10,  # >0: when a level is short by at least this many lots, add an order behind the one already there
    # round1: a fill while our side's share of the touch queue is small marks out negative (the price is
    # about to move through us); >0: rest nothing on a side whose share is below this
    'imb_pull': 0.0,
    # tried, off
    'skew_lots': 0,
    'improve_reduce': 0,
    'flat_at': 0,
    'flat_to': 0,
    # inventory study switches (defaults = ladder_mm)
    'lim_add': 100,
    'lim_flip': 100,
    'size_add': 0,
    'add_back': 0,
    'add_back_at': 50,
    'end_us': 0,
    'unwind_s': 0.0,
    'unwind_ioc_s': 0.0,
    'adapt_w_s': 0.0,
    'adapt_thr': 0.0,
    'adapt_skew': 50,
    'adapt_lim': 100,
    # walk-forward (split_data --folds / tune.py): only start trading at this backtest time (µs)
    'trade_start_us': 0,
}

BUY, SELL = Side.BUY, Side.SELL


class InvMM(BaseStrategy):
    """Rest a ladder of large orders at and behind the touch; never re-send a still-wanted order."""

    def __init__(self) -> None:
        super().__init__()
        self.p = PARAMS
        self.pos = 0
        self.orders: dict[str, list] = {}  # id -> [side, price, remaining]
        self.cancelling: set[str] = set()
        self.ioc: str | None = None
        self.unwinding = False
        self.last_ioc_us = -(10**18)
        self.cash = 0.0
        self.hist: list[tuple[int, float]] = []  # (t_us, mark-to-mid pnl) once per second
        self.adapt_on = False

    def on_private_order_update(self, event: PrivateOrderEvent) -> None:
        o = self.orders.get(event.order_id)
        if o is None:
            return
        st = event.status
        if st in (OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED):
            self.pos += event.filled_quantity if o[0] == BUY else -event.filled_quantity
            px = event.fill_price if event.fill_price is not None else o[1]
            self.cash += -px * event.filled_quantity if o[0] == BUY else px * event.filled_quantity
            o[2] = event.remaining_quantity
        elif st == OrderStatus.MODIFIED:
            o[2] = event.remaining_quantity
        if st in (OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED):
            self.orders.pop(event.order_id, None)
            self.cancelling.discard(event.order_id)
            if event.order_id == self.ioc:
                self.ioc = None

    @staticmethod
    def _touch(orders) -> tuple[int | None, int]:
        """Best market price on one side and the market quantity resting there."""
        price, qty = None, 0
        for o in orders:
            if o.owner != 'market':
                continue
            if price is None:
                price = o.price
            elif o.price != price:
                break
            qty += o.quantity
        return price, qty

    def _room(self, side: Side) -> int:
        p = self.p
        rest = sum(o[2] for oid, o in self.orders.items() if o[0] == side and oid != self.ioc)
        # signed position this side moves towards: +pos for buys, -pos for sells
        sp = self.pos if side == BUY else -self.pos
        if self.unwinding:
            lim = 0
        elif sp > 0:  # adding to the position
            lim = min(p['cap'], p['lim_add'], p['adapt_lim'] if self.adapt_on else p['cap'])
        else:  # flat or reducing: may go through flat up to lim_flip on the other side
            lim = min(p['cap'], p['lim_flip'])
        return lim - (sp + rest)

    def on_order_book_change(self, snapshot: OrderBookSnapshot, trades: list[Trade]) -> None:
        (bb, qb), (ba, qa) = self._touch(snapshot.bids), self._touch(snapshot.asks)
        p = self.p
        if bb is None or ba is None or ba - bb > p['max_spread'] or snapshot.timestamp_us < p['trade_start_us']:
            return

        now = snapshot.timestamp_us
        if p['adapt_w_s']:
            if not self.hist or now - self.hist[-1][0] >= 1_000_000:
                self.hist.append((now, self.cash + self.pos * (bb + ba) / 2))
                w = p['adapt_w_s'] * 1e6
                while len(self.hist) > 2 and now - self.hist[1][0] >= w:
                    self.hist.pop(0)
                old_t, old = self.hist[0]
                self.adapt_on = now - old_t >= w * 0.9 and self.hist[-1][1] - old < p['adapt_thr']
        if (p['end_us'] and p['unwind_s'] and not self.unwinding
                and now >= p['end_us'] - p['unwind_s'] * 1e6):
            self.unwinding = True  # drop everything once, then rest only the reducing side up to flat
            for oid in list(self.orders):
                if oid != self.ioc and oid not in self.cancelling:
                    self.cancelling.add(oid)
                    self.cancel_order(oid)
        if (p['end_us'] and p['unwind_ioc_s'] and self.pos and self.ioc is None
                and now >= p['end_us'] - p['unwind_ioc_s'] * 1e6 and now - self.last_ioc_us >= 1_000_000):
            side, px = (SELL, bb) if self.pos > 0 else (BUY, ba)
            self.unwinding = True
            self.last_ioc_us = now
            self.ioc = self.send_order(OrderRequest(side=side, price=px, quantity=abs(self.pos),
                                                    order_type=OrderType.IOC))
            self.orders[self.ioc] = [side, px, 0]

        if p['flat_at'] and self.ioc is None and abs(self.pos) >= p['flat_at']:
            side, px = (SELL, bb) if self.pos > 0 else (BUY, ba)
            self.ioc = self.send_order(OrderRequest(side=side, price=px, quantity=abs(self.pos) - p['flat_to'],
                                                    order_type=OrderType.IOC))
            self.orders[self.ioc] = [side, px, 0]

        b0, a0 = bb, ba
        if p['tight_offset'] and ba - bb == 1:
            b0, a0 = bb - 1, ba + 1
        skew = p['adapt_skew'] if self.adapt_on else p['skew_lots']
        if skew:
            shift = int(self.pos / skew)
            b0 -= max(0, shift)
            a0 += max(0, -shift)
        if p['add_back'] and abs(self.pos) >= p['add_back_at']:
            if self.pos > 0:
                b0 -= p['add_back']
            else:
                a0 += p['add_back']
        if p['improve_reduce'] and ba - bb >= 2:
            if self.pos > 0:
                a0 = ba - 1
            elif self.pos < 0:
                b0 = bb + 1

        for side, start, step, share in ((BUY, b0, -1, qb / (qb + qa)), (SELL, a0, 1, qa / (qb + qa))):
            want = [start + step * i for i in range(p['levels'])] if share >= p['imb_pull'] else []
            have: dict[int, int] = {}  # price -> resting lots of ours there
            for oid, o in self.orders.items():
                if o[0] != side or oid in self.cancelling or oid == self.ioc:
                    continue
                if o[1] in want and (p['topup'] or o[1] not in have):
                    have[o[1]] = have.get(o[1], 0) + o[2]
                else:
                    self.cancelling.add(oid)
                    self.cancel_order(oid)
            for i, price in enumerate(want):
                target = p['size_top'] if i == 0 else p['size_deep']
                if p['size_add'] and (self.pos > 0 if side == BUY else self.pos < 0):
                    target = min(target, p['size_add'])
                if price in have:
                    short = target - have[price]
                    if not p['topup'] or short < p['topup']:
                        continue
                    target = short
                q = min(target, self._room(side))
                if q <= 0:
                    continue
                oid = self.send_order(OrderRequest(side=side, price=price, quantity=q, order_type=OrderType.LIMIT))
                self.orders[oid] = [side, price, q]
