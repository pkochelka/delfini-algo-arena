"""q_ladder2: q_ladder with the research winners as defaults (levels 2, 100 at the touch + 100 one tick
behind, improve_both on). Otherwise identical; see research/round1/queue.md.

q_ladder: ladder_mm (strategies/peter/ladder_mm.py) plus queue-mechanics switches for research.

Everything defaults to off, so with default PARAMS this is ladder_mm exactly (same PnL). Research,
numbers and the ranking of these switches: research/round1/queue.md.

New switches (on top of ladder_mm's size_top / size_deep / levels / cap / max_spread / tight_offset /
topup / imb_pull / skew_lots / improve_reduce / flat_at):

    improve       when the market spread is >= improve_min_spread, step one tick inside on one side so a
                  fresh level forms with us first in its queue (never both sides: our bid and ask would meet
                  in a 2-tick spread):
                  1 thicker market touch, 2 thinner, 3 reducing side (flat: thicker),
                  4 continuation side (opposite the side whose touch last moved away),
                  5 refill side (the side whose touch last moved away), 6 whichever side has room
                  (reducing side first)
    improve_both  with a spread >= 3, step inside on both sides
    keep_behind   k > 0: an own order 1..k ticks behind the touch is kept (queue priority for when the
                  price comes back) instead of cancelled; nothing new is posted there unless levels says so
    deep_yield    when the touch order is short of size_top by >= topup and there is no room, free room by
                  shrinking own orders behind the touch, farthest first: 1 modify_order (the rest keeps its
                  priority), 2 cancel. The touch order is posted only once the shrink is confirmed, so the
                  worst-case guard always holds.
    chunk         > 0: split every new order into pieces of at most this many lots
    pre_thin      > 0: rest size_deep one tick behind the touch only while the market touch on that side
                  holds fewer than pre_thin lots (the level is about to break); levels must be 1
    pre_take      lots of the touch order's room handed to the pre_thin deep order (0 = only spare room)
    top_frac      > 0 (with levels 2): split the side's allowance A = cap -+ pos (what the guard lets that side
                  rest when nothing is filled) into touch = max(top_min, round(top_frac * A)) and deep = A - touch
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
    'size_top': 100,
    'size_deep': 100,
    'levels': 2,
    'cap': 100,
    'max_spread': 8,
    'tight_offset': 0,
    'topup': 10,
    'imb_pull': 0.0,
    'skew_lots': 0,
    'improve_reduce': 0,
    'flat_at': 0,
    'flat_to': 0,
    # queue research switches (see the docstring)
    'improve': 0,
    'improve_min_spread': 2,
    'improve_both': 1,
    'keep_behind': 0,
    'deep_yield': 0,
    'chunk': 0,
    'pre_thin': 0,
    'pre_take': 0,
    'top_frac': 0.0,
    'top_min': 0,
    'trade_start_us': 0,
}

BUY, SELL = Side.BUY, Side.SELL


class QLadder2MM(BaseStrategy):
    def __init__(self) -> None:
        super().__init__()
        self.p = PARAMS
        self.pos = 0
        self.orders: dict[str, list] = {}  # id -> [side, price, remaining]
        self.cancelling: set[str] = set()
        self.modifying: dict[str, int] = {}  # id -> requested new quantity
        self.ioc: str | None = None
        self.last_bb: int | None = None
        self.last_ba: int | None = None
        self.last_away: Side | None = None  # side whose market touch last moved away from the mid

    def on_private_order_update(self, event: PrivateOrderEvent) -> None:
        o = self.orders.get(event.order_id)
        if o is None:
            return
        st = event.status
        if st in (OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED):
            self.pos += event.filled_quantity if o[0] == BUY else -event.filled_quantity
            o[2] = event.remaining_quantity
        elif st == OrderStatus.MODIFIED:
            o[2] = event.remaining_quantity
            self.modifying.pop(event.order_id, None)
        elif st == OrderStatus.REJECTED and event.order_id in self.modifying:
            # a shrink that is no longer a shrink (fills got there first): the order lives on
            self.modifying.pop(event.order_id)
            o[2] = event.remaining_quantity
            return
        if st in (OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED):
            self.orders.pop(event.order_id, None)
            self.cancelling.discard(event.order_id)
            self.modifying.pop(event.order_id, None)
            if event.order_id == self.ioc:
                self.ioc = None

    @staticmethod
    def _touch(orders) -> tuple[int | None, int]:
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
        rest = sum(o[2] for oid, o in self.orders.items() if o[0] == side and oid != self.ioc)
        if side == BUY:
            return self.p['cap'] - (self.pos + rest)
        return self.p['cap'] + self.pos - rest

    def _send(self, side: Side, price: int, q: int) -> None:
        chunk = self.p['chunk'] or q
        while q > 0:
            n = min(chunk, q)
            oid = self.send_order(OrderRequest(side=side, price=price, quantity=n, order_type=OrderType.LIMIT))
            self.orders[oid] = [side, price, n]
            q -= n

    def on_order_book_change(self, snapshot: OrderBookSnapshot, trades: list[Trade]) -> None:
        (bb, qb), (ba, qa) = self._touch(snapshot.bids), self._touch(snapshot.asks)
        p = self.p
        if bb is not None and ba is not None:
            if self.last_bb is not None and bb < self.last_bb:
                self.last_away = BUY
            if self.last_ba is not None and ba > self.last_ba:
                self.last_away = SELL
            self.last_bb, self.last_ba = bb, ba
        if bb is None or ba is None or ba - bb > p['max_spread'] or snapshot.timestamp_us < p['trade_start_us']:
            return

        if p['flat_at'] and self.ioc is None and abs(self.pos) >= p['flat_at']:
            side, px = (SELL, bb) if self.pos > 0 else (BUY, ba)
            self.ioc = self.send_order(OrderRequest(side=side, price=px, quantity=abs(self.pos) - p['flat_to'],
                                                    order_type=OrderType.IOC))
            self.orders[self.ioc] = [side, px, 0]

        b0, a0 = bb, ba
        if p['tight_offset'] and ba - bb == 1:
            b0, a0 = bb - 1, ba + 1
        if p['skew_lots']:
            shift = int(self.pos / p['skew_lots'])
            b0 -= max(0, shift)
            a0 += max(0, -shift)
        spread = ba - bb
        if p['improve_both'] and spread >= 3:
            b0, a0 = bb + 1, ba - 1
        elif p['improve'] and spread >= p['improve_min_spread']:
            mode = p['improve']
            thick_bid = qb >= qa
            if mode == 1:
                bid_side = thick_bid
            elif mode == 2:
                bid_side = not thick_bid
            elif mode == 3:
                bid_side = self.pos < 0 or (self.pos == 0 and thick_bid)
            elif mode == 4:  # continuation: ask moved up -> the bid comes in
                bid_side = self.last_away == SELL if self.last_away is not None else thick_bid
            elif mode == 5:
                bid_side = self.last_away == BUY if self.last_away is not None else thick_bid
            else:
                rb, ra = self._room(BUY), self._room(SELL)
                bid_side = self.pos < 0 or (self.pos == 0 and rb >= ra)
            if bid_side:
                b0 = bb + 1
            else:
                a0 = ba - 1
        if p['improve_reduce'] and spread >= 2:
            if self.pos > 0:
                a0 = ba - 1
            elif self.pos < 0:
                b0 = bb + 1

        for side, start, step, mq in ((BUY, b0, -1, qb), (SELL, a0, 1, qa)):
            share = mq / (qb + qa)
            want = [start + step * i for i in range(p['levels'])] if share >= p['imb_pull'] else []
            sizes = {price: (p['size_top'] if i == 0 else p['size_deep']) for i, price in enumerate(want)}
            if want and p['top_frac']:
                allow = p['cap'] - self.pos if side == BUY else p['cap'] + self.pos
                top = min(allow, max(p['top_min'], round(p['top_frac'] * allow)))
                sizes = {price: (top if i == 0 else max(0, allow - top)) for i, price in enumerate(want)}
            if want and p['pre_thin'] and mq < p['pre_thin']:
                want.append(start + step)
                sizes[start + step] = p['size_deep']
                if p['pre_take']:
                    sizes[start] = max(0, sizes[start] - p['pre_take'])
            keep = set()
            if want and p['keep_behind']:
                keep = {start + step * k for k in range(1, p['keep_behind'] + 1)} - set(want)
            have: dict[int, int] = {}
            behind: list[tuple[int, str]] = []  # (ticks behind start, id) of own orders behind the touch
            for oid, o in self.orders.items():
                if o[0] != side or oid in self.cancelling or oid == self.ioc:
                    continue
                if o[1] in keep:
                    behind.append(((o[1] - start) * step, oid))
                    continue
                if o[1] in want and (p['topup'] or o[1] not in have):
                    have[o[1]] = have.get(o[1], 0) + o[2]
                    if o[1] != start:
                        behind.append(((o[1] - start) * step, oid))
                else:
                    self.cancelling.add(oid)
                    self.cancel_order(oid)
            for price in want:
                target = sizes[price]
                if target <= 0:
                    continue
                if price in have:
                    short = target - have[price]
                    if not p['topup'] or short < p['topup']:
                        continue
                    target = short
                room = self._room(side)
                if price == start and p['deep_yield'] and room < target and behind:
                    need = target - max(room, 0)
                    for _, oid in sorted(behind, reverse=True):
                        if need <= 0:
                            break
                        if oid in self.modifying or oid in self.cancelling:
                            need -= self.orders[oid][2] - self.modifying.get(oid, 0)
                            continue
                        rem = self.orders[oid][2]
                        if p['deep_yield'] == 1 and rem > need:
                            self.modifying[oid] = rem - need
                            self.modify_order(oid, rem - need)
                            need = 0
                        else:
                            self.cancelling.add(oid)
                            self.cancel_order(oid)
                            need -= rem
                    behind = []
                q = min(target, room)
                if q <= 0:
                    continue
                self._send(side, price, q)
