"""queue_mm: ladder_mm's queue-seniority market maker, made to adapt to the market's price grid and book depth.

Round1 fill markouts (see queue_hypotheses.md): the edge is queue seniority. Fills of orders at the front
of a queue / older than 0.5 s mark out about +0.35 ticks at 1 s, fresh back-of-queue ones about +0.1, and
the first order in a brand-new level is toxic. So: rest at the touch (and one level behind), never
re-send a still-wanted order, and avoid any action that resets queue position (pulls, stepping inside the spread).

Per side, the wanted prices are the touch (best market price, our own orders excluded) and `levels - 1`
grid steps behind it. The grid step is learned from the book (most common gap between occupied levels),
so on a coarser tick grid the deep order joins a real level instead of founding a fresh one. When the
market touch is thick relative to our cap (EWMA touch > deep_ratio * cap), a deep order only takes room
from the touch: rest one level of `thick_size` lots instead. A guard keeps the worst case
|position + every open order on a side| within `cap`, so the +-100 penalty can't trigger with cap <= 100.

Tried and kept behind flags (all hurt, see queue_hypotheses.md): imb_pull, vanish_qty, improve,
improve_reduce, size_k, gate_share, tight_offset, flat_at/flat_to.
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
    'size_top': 99,  # lots at the touch (clipped by the guard; 200 = cap + a full opposite position)
    'size_deep': 50,  # lots at each level behind the touch
    'levels': 2,  # price levels per side, touch included
    'cap': 100,  # worst-case |position + open orders on a side|
    'max_spread': 8,  # ignore books wider than this (a side backed by stale far orders)
    'tight_offset': 0,  # 1: with a 1-tick spread, start the ladder one tick behind the touch
    'topup': 20,  # >0: when a level is short by at least this many lots, add an order behind the one already there
    # round1: a fill while our side's share of the touch queue is small marks out negative (the price is
    # about to move through us); >0: rest nothing on a side whose share is below this
    'imb_pull': 0.0,
    # >0: cancel just the touch order on a side whose market touch qty is below this while the other
    # touch holds >= vanish_ratio times as much (deeper levels keep their priority)
    'vanish_qty': 0,
    'vanish_ratio': 10.0,
    # market price grid in ticks (levels behind the touch, skew and max_spread are in grid steps);
    # 0 = learn it from the first grid_learn snapshots (most common gap between occupied levels)
    'grid_unit': 0,
    # >0: touch size = clip(size_k * EWMA market taker size, size_top, size_top_max) (thick books want more)
    'size_k': 0.0,
    # >0: rest only at the touch (levels = 1) while the EWMA market touch qty exceeds deep_ratio * cap
    'deep_ratio': 18.0,
    'touch_n': 20000,
    'thick_size': 200,  # >0: size_top while deep_ratio has switched to one level
    'size_top_max': 200,
    'size_n': 1000,
    'grid_learn': 5000,
    'gate_share': 0.0,  # >0: send no new touch order on a side whose market touch share is below this
    'gate_all': 0,  # 1: the gate also blocks new deep orders
    'improve': 0,  # see on_order_book_change: 1 thick side, 2 thin side, 3 reducing side
    # tried, off
    'skew_lots': 100,
    'improve_reduce': 0,
    'flat_at': 0,
    'flat_to': 0,
    # walk-forward (split_data --folds / tune.py): only start trading at this backtest time (µs)
    'trade_start_us': 0,
}

BUY, SELL = Side.BUY, Side.SELL


class QueueMM(BaseStrategy):
    """Rest a ladder of large orders at and behind the touch; never re-send a still-wanted order."""

    def __init__(self) -> None:
        super().__init__()
        self.p = PARAMS
        self.pos = 0
        self.orders: dict[str, list] = {}  # id -> [side, price, remaining]
        self.cancelling: set[str] = set()
        self.ioc: str | None = None
        self.unit = max(1, self.p['grid_unit'])  # price grid of the market's levels, in ticks
        self.gaps: dict[int, int] = {}  # gap between adjacent occupied market levels -> count
        self.seen = 0
        self.aggr = 0.0  # EWMA of a market taker order's traded quantity
        self.n_aggr = 0
        self.touch_q = 0.0  # EWMA of market touch qty per side
        self.n_touch = 0

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
        rest = sum(o[2] for oid, o in self.orders.items() if o[0] == side and oid != self.ioc)
        if side == BUY:
            return self.p['cap'] - (self.pos + rest)
        return self.p['cap'] + self.pos - rest

    def _learn_unit(self, snapshot: OrderBookSnapshot) -> None:
        """Most common gap between adjacent occupied market price levels (both sides, top 4 each)."""
        for orders in (snapshot.bids, snapshot.asks):
            last, n = None, 0
            for o in orders:
                if o.owner != 'market' or o.price == last:
                    continue
                if last is not None:
                    g = abs(o.price - last)
                    self.gaps[g] = self.gaps.get(g, 0) + 1
                last, n = o.price, n + 1
                if n >= 4:
                    break
        self.seen += 1
        if self.seen % 500 == 0:
            self.unit = max(self.gaps, key=self.gaps.get)

    def on_order_book_change(self, snapshot: OrderBookSnapshot, trades: list[Trade]) -> None:
        p = self.p
        if not p['grid_unit'] and self.seen < p['grid_learn']:
            self._learn_unit(snapshot)
        if trades and p['size_k']:
            per: dict[str, int] = {}
            for tr in trades:
                if tr.taker_owner == 'market':
                    per[tr.taker_order_id] = per.get(tr.taker_order_id, 0) + tr.quantity
            for q in per.values():
                self.n_aggr += 1
                a = max(1.0 / self.n_aggr, 1.0 / p['size_n'])
                self.aggr += a * (q - self.aggr)
        (bb, qb), (ba, qa) = self._touch(snapshot.bids), self._touch(snapshot.asks)
        u = self.unit
        if p['deep_ratio']:
            self.n_touch += 1
            a = max(1.0 / self.n_touch, 1.0 / p['touch_n'])
            self.touch_q += a * ((qb + qa) / 2 - self.touch_q)
        levels = p['levels']
        if p['deep_ratio'] and self.touch_q > p['deep_ratio'] * p['cap']:
            levels = 1  # our capacity is a small slice of a thick queue: a deep order only takes room from the touch
        size_top = p['thick_size'] if levels == 1 and p['thick_size'] and p['deep_ratio'] else p['size_top']
        if p['size_k'] and self.n_aggr >= 50:
            size_top = min(p['size_top_max'], max(size_top, round(p['size_k'] * self.aggr)))
        if bb is None or ba is None or ba - bb > p['max_spread'] * u or snapshot.timestamp_us < p['trade_start_us']:
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
            b0 -= max(0, shift) * u
            a0 += max(0, -shift) * u
        if p['improve'] and ba - bb >= 2:
            # step inside a wide spread on one side to be first in a fresh level's queue:
            # 1 = the side with the thicker market touch, 2 = the thinner one, 3 = the inventory-reducing one
            # (flat: thicker); never both, so our own bid and ask can't meet
            thick_bid = qb >= qa
            if p['improve'] == 1:
                bid_side = thick_bid
            elif p['improve'] == 2:
                bid_side = not thick_bid
            else:
                bid_side = self.pos < 0 or (self.pos == 0 and thick_bid)
            if bid_side:
                b0 = bb + 1
            else:
                a0 = ba - 1
        if p['improve_reduce'] and ba - bb >= 2:
            if self.pos > 0:
                a0 = ba - 1
            elif self.pos < 0:
                b0 = bb + 1

        for side, start, step, share, mine, other in ((BUY, b0, -u, qb / (qb + qa), qb, qa),
                                                      (SELL, a0, u, qa / (qb + qa), qa, qb)):
            want = [start + step * i for i in range(levels)] if share >= p['imb_pull'] else []
            # drop only the touch price when its market queue is nearly gone and the other side is thick:
            # the level is about to vanish, so its queue priority is worth little and its fills mark out badly
            if want and p['vanish_qty'] and mine < p['vanish_qty'] and other >= p['vanish_ratio'] * max(mine, 1):
                want = want[1:]
            have: dict[int, int] = {}  # price -> resting lots of ours there
            for oid, o in self.orders.items():
                if o[0] != side or oid in self.cancelling or oid == self.ioc:
                    continue
                if o[1] in want and (p['topup'] or o[1] not in have):
                    have[o[1]] = have.get(o[1], 0) + o[2]
                else:
                    self.cancelling.add(oid)
                    self.cancel_order(oid)
            # no new orders (existing ones keep their queue spot) on a side whose market touch share is thin:
            # it is about to be run over, and a fresh order there joins at the back of a dying queue
            gate = p['gate_share'] and share < p['gate_share']
            for price in want:
                if gate and (price == start or p['gate_all']):
                    continue
                target = size_top if price == start else p['size_deep']
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
