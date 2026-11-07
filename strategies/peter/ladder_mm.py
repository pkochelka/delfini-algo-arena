"""ladder_mm: passive market maker that rests large orders at and behind the touch and keeps queue priority.

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
    # tried, off
    'skew_lots': 0,
    'improve_reduce': 0,
    'flat_at': 0,
    'flat_to': 0,
}

BUY, SELL = Side.BUY, Side.SELL


class LadderMM(BaseStrategy):
    """Rest a ladder of large orders at and behind the touch; never re-send a still-wanted order."""

    def __init__(self) -> None:
        super().__init__()
        self.p = PARAMS
        self.pos = 0
        self.orders: dict[str, list] = {}  # id -> [side, price, remaining]
        self.cancelling: set[str] = set()
        self.ioc: str | None = None

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
    def _touch(orders) -> int | None:
        for o in orders:
            if o.owner == 'market':
                return o.price
        return None

    def _room(self, side: Side) -> int:
        rest = sum(o[2] for oid, o in self.orders.items() if o[0] == side and oid != self.ioc)
        if side == BUY:
            return self.p['cap'] - (self.pos + rest)
        return self.p['cap'] + self.pos - rest

    def on_order_book_change(self, snapshot: OrderBookSnapshot, trades: list[Trade]) -> None:
        bb, ba = self._touch(snapshot.bids), self._touch(snapshot.asks)
        p = self.p
        if bb is None or ba is None or ba - bb > p['max_spread']:
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
        if p['improve_reduce'] and ba - bb >= 2:
            if self.pos > 0:
                a0 = ba - 1
            elif self.pos < 0:
                b0 = bb + 1

        for side, start, step in ((BUY, b0, -1), (SELL, a0, 1)):
            want = [start + step * i for i in range(p['levels'])]
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
