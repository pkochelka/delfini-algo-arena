"""impact_mm: touch market maker that steps out of the way of large aggressive orders' delayed impact.

Research (round1, every aggressive order, mid move in the order's direction vs the pre-trade mid,
ticks; scratchpad script, numbers reproduced by `analysis.features.taker_orders` + `asof`):

    order size     n       5 ms    100 ms   1 s     5 s
    <= 50       42,798    0.025   0.046   0.134   0.142
    50-100      18,236    0.051   0.097   0.246   0.237
    100-200     15,594    0.080   0.137   0.341   0.349
    200-400     10,353    0.130   0.215   0.467   0.465
    400-800      5,246    0.216   0.353   0.678   0.631
    800-1600     2,028    0.390   0.594   0.951   0.988
    > 1600         442    0.567   0.900   1.421   1.538

The impact grows monotonically with size and roughly half of it arrives after our 5 ms latency,
over 50-1000 ms, then stays (no reversion by 5 s). Taking on it does not pay: an IOC landing at
t + 5 ms buys at an ask that has already moved, and entry-vs-mid-later is negative for every
size bucket at 0.25-2 s (-0.35..-0.50 ticks), only > 1600 lots turns positive at 5 s (+0.09),
before paying the spread to get out. So the edge is used passively:

  * a decayed signed aggressive-flow signal S (lots) is updated from the public trades:
        S <- S * exp(-dt / tau) + sum(sign * qty) of new market aggressive orders
  * |S| >= `guard_lots`: the side the flow is about to run over (asks after buying, bids after
    selling) is pulled (`guard_mode` = 'pull'), moved `guard_back` ticks behind the touch
    ('back') or frozen (orders kept with their queue priority, nothing new posted: 'freeze'),
    until |S| decays below `guard_lots * guard_release`. The other side keeps quoting.
  * optional `cover_lots`: when |S| >= cover_lots and our inventory is on the wrong side of the
    flow, IOC `cover_frac` of it at the touch (pays half a spread, avoids the remaining impact).

Core quoting is ladder_mm (strategies/peter/ladder_mm.py): big resting orders at the touch,
queue priority kept, worst case |position + open orders on a side| <= `cap`.

Ladder fills on wf01, size-weighted markout (mid later - fill price, our direction, ticks):

    filled BY an order of     lots     1 s     5 s    30 s
    <= 100                   71,316   +0.37   +0.35   +0.51
    100-300                  98,225   +0.24   +0.18   -0.09
    300-800                  60,099   +0.04   +0.11   +0.05
    800-1600                 19,648   -0.37   -0.36   +0.35
    > 1600                    4,723   -0.82   -0.84   +0.29
    prior flow vs us 100-600 21,774   -0.22   -0.14   +0.06   (what the guard can avoid)

Results, round1 walk-forward folds wf01-10 (strategies/marek/sweep_impact.py, total PnL):

    variant                         cap 100   cap 60    cap 30
    no guard (= ladder_mm)          425,118   331,805   207,194
    pull 600 / tau 300              420,860   324,801   199,683
    pull 300 / tau 100              416,384   324,389   200,444
    freeze 600 / tau 300            408,823   318,192   196,821
    pull 600 / tau 50               426,311                       (+1.2k, fold sd 2.5k: noise)
    pull 1000 / 1500, tau 300       422,085 / 424,433
    pull 600 / tau 1000             364,046
    back 1 tick, 600 / tau 300      424,909
    pull, only orders >= 800        417,144
    cover 400 / 800 / 1600 (IOC)    276,947 / 364,910 / 414,892

Conclusion: the delayed impact is real and large, but nothing here monetizes it. The fills it
hurts are mostly the ones caused by the big order itself, which no post-trade signal can avoid;
the avoidable ones (after flow) lose at 1 s but recover by 30 s, which is the horizon a ladder
holding inventory is actually scored on, and pulling or freezing a side costs queue priority
and good fills. Covering pays half a spread to dodge ~0.5 tick of move, which is a wash at best.
So the overlays default to off (guard = 0, cover_lots = 0) and the strategy is the ladder;
they stay as switches for later rounds, where the impact/spread ratio may differ.

Queue position (ladder fills on wf01/04/07 by market lots ahead of us at the fill, 30 s markout):
ahead 0: 410k lots +0.20; 1-200: 328k +0.15; 200-500: 31k +0.09; 500-1000: 8k -0.48;
1000-2000: 2k -1.65. `max_join` (only post where < N market lots rest) still loses money at
every N (100: 190,887 / 300: 265,700 / 600: 335,561 / 1000: 385,160 / 2000: 414,344 vs 425,118):
orders that join at the back reach the front later, and the resting size is what earns. Turning
off the top-up costs even more (topup 0: 259,853). Loosening `cap` above 100 (resting orders
beyond the limit, IOC back to 95 on a breach) loses over 1M to breach penalties.

Every overlay is a PARAMS switch so it can be turned off for A/B and tuned with
strategies/peter/tune.py (add an `impact_mm` space there).
"""

from __future__ import annotations

import math

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
    # ladder core
    'size_top': 200,  # lots at the touch (clipped by the cap guard)
    'cap': 100,  # worst-case |position + open orders on a side|
    'max_spread': 8,  # ignore books wider than this
    'topup': 10,  # >0: add behind our order when a level is short by at least this many lots
    'max_join': 0,  # >0: only post a new order where fewer market lots than this already rest (front of queue)
    # impact signal
    'tau_ms': 300.0,  # decay of the signed aggressive-flow signal
    'min_order': 0,  # ignore aggressive orders smaller than this (lots)
    # guard: stop quoting the side that will be run over
    'guard': 0,  # off: costs 1-16k on round1 at every setting tried (see docstring)
    'guard_lots': 600.0,
    'guard_release': 0.5,  # leave the alert when |S| < guard_lots * guard_release
    'guard_mode': 'pull',  # 'pull' (cancel), 'back' (re-quote behind) or 'freeze' (keep, add nothing)
    'guard_back': 1,  # ticks behind the touch in 'back' mode
    # cover: IOC out of inventory that sits against strong flow
    'cover_lots': 0.0,  # 0 = off
    'cover_frac': 1.0,
    'cover_cooldown_ms': 200.0,
    # walk-forward (tune.py): only trade from this backtest time (µs)
    'trade_start_us': 0,
}

BUY, SELL = Side.BUY, Side.SELL


class ImpactMM(BaseStrategy):
    """Ladder market maker with a large-order impact guard."""

    def __init__(self) -> None:
        super().__init__()
        self.p = PARAMS
        self.pos = 0
        self.orders: dict[str, list] = {}  # id -> [side, price, remaining, is_ioc]
        self.cancelling: set[str] = set()
        self.flow = 0.0  # decayed signed aggressive volume, lots (+ = buying)
        self.flow_t = 0
        self.alert = 0  # +1: buy pressure (asks unsafe), -1: sell pressure (bids unsafe), 0: none
        self.last_cover_us = -(10**18)

    # ------------------------------------------------------------------ feeds

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

    def on_order_book_change(self, snapshot: OrderBookSnapshot, trades: list[Trade]) -> None:
        now = snapshot.timestamp_us
        self._update_flow(now, trades)
        p = self.p
        bb, ba = _touch(snapshot.bids), _touch(snapshot.asks)
        if bb is None or ba is None or ba - bb > p['max_spread'] or now < p['trade_start_us']:
            return

        if p['cover_lots'] > 0:
            self._maybe_cover(now, bb, ba)

        targets = {BUY: bb, SELL: ba}
        frozen = None
        if p['guard'] and self.alert:
            unsafe = SELL if self.alert > 0 else BUY
            if p['guard_mode'] == 'freeze':
                frozen = unsafe
            elif p['guard_mode'] == 'pull':
                targets[unsafe] = None
            else:
                targets[unsafe] += p['guard_back'] if unsafe == SELL else -p['guard_back']

        self.snapshot = snapshot
        for side in (BUY, SELL):
            if side != frozen:
                self._quote(side, targets[side])

    # ----------------------------------------------------------------- signal

    def _update_flow(self, now: int, trades: list[Trade]) -> None:
        p = self.p
        if now != self.flow_t:
            self.flow *= math.exp(-(now - self.flow_t) / (p['tau_ms'] * 1000.0))
            self.flow_t = now
        if trades:
            by_taker: dict[str, int] = {}
            for tr in trades:
                if tr.taker_owner != 'market':
                    continue
                s = 1 if tr.book_side == 'ask' else -1
                by_taker[tr.taker_order_id] = by_taker.get(tr.taker_order_id, 0) + s * tr.quantity
            for q in by_taker.values():
                if abs(q) >= p['min_order']:
                    self.flow += q
        on, off = p['guard_lots'], p['guard_lots'] * p['guard_release']
        if self.flow >= on:
            self.alert = 1
        elif self.flow <= -on:
            self.alert = -1
        elif abs(self.flow) < off:
            self.alert = 0

    # ---------------------------------------------------------------- trading

    def _maybe_cover(self, now: int, bb: int, ba: int) -> None:
        p = self.p
        f = self.flow
        if abs(f) < p['cover_lots'] or now - self.last_cover_us < p['cover_cooldown_ms'] * 1000:
            return
        if any(o[3] for o in self.orders.values()):
            return
        # Long into selling or short into buying.
        if f < 0 and self.pos > 0:
            side, px = SELL, bb
        elif f > 0 and self.pos < 0:
            side, px = BUY, ba
        else:
            return
        q = min(int(abs(self.pos) * p['cover_frac']), self._room(side))
        if q <= 0:
            return
        oid = self.send_order(OrderRequest(side=side, price=px, quantity=q, order_type=OrderType.IOC))
        self.orders[oid] = [side, px, q, True]
        self.last_cover_us = now

    def _room(self, side: Side) -> int:
        rest = sum(o[2] for o in self.orders.values() if o[0] == side)
        if side == BUY:
            return self.p['cap'] - (self.pos + rest)
        return self.p['cap'] + self.pos - rest

    def _quote(self, side: Side, price: int | None) -> None:
        p = self.p
        have = 0
        for oid, o in self.orders.items():
            if o[0] != side or o[3] or oid in self.cancelling:
                continue
            if price is not None and o[1] == price:
                have += o[2]
            else:
                self.cancelling.add(oid)
                self.cancel_order(oid)
        if price is None or price <= 0:
            return
        if p['max_join'] and _market_qty(self.snapshot, side, price) >= p['max_join']:
            return  # would join at the back of a long queue: only sweeps reach us there
        target = p['size_top']
        if have:
            short = target - have
            if not p['topup'] or short < p['topup']:
                return
            target = short
        q = min(target, self._room(side))
        if q <= 0:
            return
        oid = self.send_order(OrderRequest(side=side, price=price, quantity=q, order_type=OrderType.LIMIT))
        self.orders[oid] = [side, price, q, False]


def _market_qty(snapshot: OrderBookSnapshot, side: Side, price: int) -> int:
    book = snapshot.bids if side == BUY else snapshot.asks
    return sum(o.quantity for o in book if o.price == price and o.owner == 'market')


def _touch(orders) -> int | None:
    for o in orders:
        if o.owner == 'market':
            return o.price
    return None
