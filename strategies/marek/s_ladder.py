"""s_ladder: sig_ladder's queue-priority ladder driven by the hidden fair price S (round1 research).

Round1 is generated from one hidden price (strategies/peter/round1_generator.md):
    S = S0 + c * sum over taker IOCs of sign * sqrt(filled),   c = 0.026
Makers quote bid <= S <= ask and cancel stale quotes only 20-40 ms after S moves; we see the
taker's trade immediately and act after 5 ms. Splitting sig_ladder's fills by our price's distance
from S (dist = S - bid for a bid, ask - S for an ask) before the fill gives the edge per lot after
the fill's own jump: dist < 0 -> -0.51, 0-0.25 -> -0.25, 0.25-0.5 -> -0.04, 0.5-0.75 -> +0.24,
0.75-1 -> +0.47, 1-1.25 -> +0.70, 1.25-1.5 -> +0.93. Fills with dist < 0.5 were 38% of lots and lost
~137k. On a 1-tick spread the two sides' dists sum to 1, so one side is always >= 0.5.

S estimate (live): S_raw += c * sqrt(total filled by a market taker in this event) with the sign of
the aggressor. Our own resting orders change how much a taker fills relative to the tape, so S_raw
drifts (~6 ticks sd over the tape). It is corrected by `off` so that every new market maker order seen
at the touch satisfies bid <= S - s_gap, ask >= S + s_gap (generator: price never within 0.05 of S).
Offline (sig_ladder's run): error sd 0.12 tick, p1/p99 -0.33/+0.32 against the tape's exact S.

Mechanics (each behind a PARAMS key; s_pull 0, s_take 0 and sig guard on = sig_ladder):
    s_pull    a side is "bad" while dist < s_pull (released at s_pull + s_hyst). While bad it is handled
              like sig_ladder's unsafe side: shrink the touch order to the flattening part (s_keep_flat) or
              cancel it, and pre-join one tick behind.
    s_take    IOC-snipe: buy every ask level <= S - s_take (sell every bid >= S + s_take), up to the room
              left under the cap. Stale quotes after a jump in S; S has no drift so the edge is kept.
    s_improve with a spread >= 2 ticks, quote inside at the best price p with S - p >= s_improve (bid).
    use_sig   1: keep sig_ladder's depletion/flow guard as well (side unsafe if either says so).
"""

from __future__ import annotations

import math
from collections import deque

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
    # ladder core (ladder_mm defaults)
    'size_top': 200,
    'cap': 100,
    'max_spread': 8,
    'topup': 10,
    'trade_start_us': 0,
    # hidden price S
    's_c': 0.026,
    's_exp': 0.5,
    's_gap': 0.05,  # makers never quote within this of S
    # S-driven quoting
    's_pull': 0.3,  # 0 = off
    's_hyst': 0.0,
    's_pull_flat': 0.0,  # the flattening part is pulled too once dist < this (-9 = never)
    's_pull_pos': 0.0,  # adding-side threshold = s_pull + s_pull_pos * |pos| / cap
    's_keep_flat': 1,  # while bad keep min(resting, |pos|) at the touch on the side that flattens
    's_prejoin': 1,  # while bad rest the freed room one tick behind the touch
    's_gen': 1,  # 1: quote each side at the best price p with dist(p) >= threshold (the pre-joined order follows S)
    's_maxdepth': 3,  # s_gen: don't rest more than this many ticks behind the touch
    's_take': 0.1,  # 0 = off; IOC when the far touch is at least this far through S
    's_take_max': 200,
    's_improve': 0.4,  # 0 = off
    # sig_ladder guard (defaults = sig_ladder's)
    's_oracle': '',  # research only: path to an npz of the tape's exact S (t, S); never in a submission
    'use_sig': 0,
    'g_share': 0.10,
    'g_dep': 0.5,
    'flow_k': 0.10,
    'flow_ms': 1000,
    'flow2_k': 0.5,
    'flow2_ms': 200,
    'improve_both': 1,
}

BUY, SELL = Side.BUY, Side.SELL


class SLadder(BaseStrategy):
    def __init__(self) -> None:
        super().__init__()
        self.p = PARAMS
        self.pos = 0
        self.orders: dict[str, list] = {}  # id -> [side, price, remaining, seq, is_ioc]
        self.cancelling: set[str] = set()
        self.modifying: dict[str, int] = {}
        self.mod_n: dict[str, int] = {}
        self.seq = 0
        self.ioc_live = {BUY: 0, SELL: 0}
        # S
        self.s_raw = None  # set from the first two-sided book
        self.off = 0.0
        self.max_id = 0
        self.last_touch_px = {BUY: None, SELL: None}
        # guards
        self.bad = {BUY: False, SELL: False}
        self.unsafe = {BUY: False, SELL: False}
        self.flow: deque = deque()
        self.flow_sum = 0
        self.peak = {BUY: (None, 0), SELL: (None, 0)}

    # ------------------------------------------------------------------ private feed
    def on_private_order_update(self, event: PrivateOrderEvent) -> None:
        oid = event.order_id
        o = self.orders.get(oid)
        if o is None:
            return
        st = event.status
        if st in (OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED):
            self.pos += event.filled_quantity if o[0] == BUY else -event.filled_quantity
            o[2] = event.remaining_quantity
        elif st == OrderStatus.MODIFIED:
            o[2] = event.remaining_quantity
            self._mod_done(oid)
        if st == OrderStatus.REJECTED and self.mod_n.get(oid):
            self._mod_done(oid)
            if event.remaining_quantity:
                o[2] = event.remaining_quantity
            return
        if st in (OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED):
            self.orders.pop(oid, None)
            self.cancelling.discard(oid)
            self.modifying.pop(oid, None)
            self.mod_n.pop(oid, None)
            if o[4]:
                self.ioc_live[o[0]] -= 1

    def _mod_done(self, oid: str) -> None:
        n = self.mod_n.get(oid, 0) - 1
        if n <= 0:
            self.mod_n.pop(oid, None)
            self.modifying.pop(oid, None)
        else:
            self.mod_n[oid] = n

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _touch(orders):
        """(price, market qty, last market order id at the touch)."""
        price, qty, last = None, 0, None
        for o in orders:
            if o.owner != 'market':
                continue
            if price is None:
                price = o.price
            elif o.price != price:
                break
            qty += o.quantity
            last = o.order_id
        return price, qty, last

    def _room(self, side: Side) -> int:
        rest = sum(o[2] for o in self.orders.values() if o[0] == side)
        if side == BUY:
            return self.p['cap'] - (self.pos + rest)
        return self.p['cap'] + self.pos - rest

    def _send(self, side: Side, price: int, q: int, ioc: bool = False) -> None:
        oid = self.send_order(OrderRequest(side=side, price=price, quantity=q,
                                           order_type=OrderType.IOC if ioc else OrderType.LIMIT))
        self.seq += 1
        self.orders[oid] = [side, price, q, self.seq, ioc]
        if ioc:
            self.ioc_live[side] += 1

    def _cancel(self, oid: str) -> None:
        self.cancelling.add(oid)
        self.cancel_order(oid)

    def _shrink(self, oid: str, o: list, keep: int) -> int:
        """Shrink order to at most `keep` (cancel if keep <= 0); return what is left of keep."""
        cur = min(o[2], self.modifying.get(oid, o[2]))
        if keep <= 0:
            self._cancel(oid)
            return 0
        if cur > keep:
            self.modifying[oid] = keep
            self.mod_n[oid] = self.mod_n.get(oid, 0) + 1
            self.modify_order(oid, keep)
            return 0
        return keep - cur

    # ------------------------------------------------------------------ S
    def _update_s(self, trades: list[Trade]) -> None:
        p = self.p
        per: dict[str, list] = {}
        for tr in trades:
            if tr.taker_owner != 'market':
                continue
            e = per.get(tr.taker_order_id)
            if e is None:
                per[tr.taker_order_id] = [tr.quantity, 1 if tr.book_side == 'ask' else -1]
            else:
                e[0] += tr.quantity
        if self.s_raw is not None:
            for q, sg in per.values():
                self.s_raw += sg * p['s_c'] * q ** p['s_exp']

    def _bound_s(self, side: Side, px: int | None, last: str | None) -> None:
        """A new market maker order at the touch bounds S: bid <= S - gap, ask >= S + gap."""
        prev = self.last_touch_px[side]
        self.last_touch_px[side] = px
        if last is None or self.s_raw is None:
            return
        try:
            i = int(last)
        except ValueError:
            return
        if i <= self.max_id:
            return
        self.max_id = i
        # an old order surfacing when the touch retreats is not a new insert
        if prev is not None and ((side == BUY and px < prev) or (side == SELL and px > prev)):
            return
        s = self.s_raw + self.off
        g = self.p['s_gap']
        if side == BUY and s < px + g:
            self.off += px + g - s
        elif side == SELL and s > px - g:
            self.off -= s - (px - g)

    # ------------------------------------------------------------------ main
    def on_order_book_change(self, snapshot: OrderBookSnapshot, trades: list[Trade]) -> None:
        p = self.p
        now = snapshot.timestamp_us
        self._update_s(trades)
        if p['use_sig'] and (p['flow_k'] or p['flow2_k']):
            for tr in trades:
                if tr.taker_owner == 'market':
                    s = tr.quantity if tr.book_side == 'ask' else -tr.quantity
                    self.flow.append((now, s))
                    self.flow_sum += s
            lim = now - p['flow_ms'] * 1000
            while self.flow and self.flow[0][0] < lim:
                self.flow_sum -= self.flow.popleft()[1]

        (bb, qb, lb), (ba, qa, la) = self._touch(snapshot.bids), self._touch(snapshot.asks)
        if bb is None or ba is None:
            return
        if self.s_raw is None:
            self.s_raw = (bb + ba) / 2.0
        self._bound_s(BUY, bb, lb)
        self._bound_s(SELL, ba, la)
        S = self.s_raw + self.off
        if p['s_oracle']:
            S = self._oracle(now)
        self.S = S
        if ba - bb > p['max_spread'] or now < p['trade_start_us']:
            return

        # ---- snipe stale quotes through S
        if p['s_take']:
            self._snipe(snapshot, S)

        # ---- S rule: bad side
        self.sev = {BUY: False, SELL: False}
        if p['s_pull']:
            for side, d in ((BUY, S - bb), (SELL, ba - S)):
                flat = (side == SELL and self.pos > 0) or (side == BUY and self.pos < 0)
                th = p['s_pull'] if flat else p['s_pull'] + p['s_pull_pos'] * abs(self.pos) / p['cap']
                if d < th:
                    self.bad[side] = True
                elif d >= th + p['s_hyst']:
                    self.bad[side] = False
                self.sev[side] = d < p['s_pull_flat']

        # ---- sig_ladder guard
        if p['use_sig']:
            self._sig_guard(now, bb, qb, ba, qa)

        qb_px, qa_px = bb, ba
        if p['improve_both'] and ba - bb >= 3:
            qb_px, qa_px = bb + 1, ba - 1
        if p['s_improve'] and ba - bb >= 2:
            tb = min(ba - 1, math.floor(S - p['s_improve']))
            ta = max(bb + 1, math.ceil(S + p['s_improve']))
            if tb > qb_px:
                qb_px = tb
            if ta < qa_px:
                qa_px = ta

        if p['s_gen']:
            self._quote_gen(S, bb, ba, qb_px, qa_px)
            return
        for side, touch, step in ((BUY, qb_px, -1), (SELL, qa_px, 1)):
            unsafe = self.bad[side] or self.unsafe[side]
            mine = sorted(((o[3], oid, o) for oid, o in self.orders.items()
                           if o[0] == side and not o[4] and oid not in self.cancelling), key=lambda x: x[0])
            at_touch, behind = [], []
            for _, oid, o in mine:
                d = (touch - o[1]) if side == BUY else (o[1] - touch)
                if d == 0:
                    at_touch.append((oid, o))
                elif d == 1 and unsafe:
                    behind.append((oid, o))
                else:
                    self._cancel(oid)
            if unsafe:
                keep = 0
                if p['s_keep_flat'] and not self.sev[side] and ((side == SELL and self.pos > 0) or (side == BUY and self.pos < 0)):
                    keep = abs(self.pos)
                for oid, o in at_touch:
                    keep = self._shrink(oid, o, keep)
                if p['s_prejoin'] and not behind:
                    q = min(p['size_top'], self._room(side))
                    if q > 0 and touch + step > 0:
                        self._send(side, touch + step, q)
                continue
            have = sum(o[2] for _, o in at_touch)
            target = p['size_top']
            if have:
                short = target - have
                target = short if p['topup'] and short >= p['topup'] else 0
            q = min(target, self._room(side))
            if q > 0:
                self._send(side, touch, q)

    def _quote_gen(self, S: float, bb: int, ba: int, qb_px: int, qa_px: int) -> None:
        p = self.p
        for side, base, mt, step in ((BUY, qb_px, bb, -1), (SELL, qa_px, ba, 1)):
            flat = (side == SELL and self.pos > 0) or (side == BUY and self.pos < 0)
            th = p['s_pull'] if flat else p['s_pull'] + p['s_pull_pos'] * abs(self.pos) / p['cap']
            if side == BUY:
                tgt = base if S - base >= th else min(base - 1, math.floor(S - th))
            else:
                tgt = base if base - S >= th else max(base + 1, math.ceil(S + th))
            good = tgt == base
            mine = sorted(((o[3], oid, o) for oid, o in self.orders.items()
                           if o[0] == side and not o[4] and oid not in self.cancelling), key=lambda x: x[0])
            at, ahead = [], []
            for _, oid, o in mine:
                d = (tgt - o[1]) * -step  # >0: o is more aggressive than tgt
                d = (o[1] - tgt) if side == BUY else (tgt - o[1])
                if d == 0:
                    at.append((oid, o))
                elif d > 0 and not good:
                    ahead.append((oid, o))
                else:
                    self._cancel(oid)
            if not good:
                keep = abs(self.pos) if (p['s_keep_flat'] and flat and not self.sev[side]) else 0
                for oid, o in ahead:
                    keep = self._shrink(oid, o, keep)
                if not p['s_prejoin'] or abs(tgt - mt) > p['s_maxdepth'] or tgt <= 0:
                    for oid, o in at:
                        self._cancel(oid)
                    continue
            have = sum(o[2] for _, o in at)
            target = p['size_top']
            if have:
                short = target - have
                target = short if p['topup'] and short >= p['topup'] and good else 0
            q = min(target, self._room(side))
            if q > 0:
                self._send(side, tgt, q)

    def _oracle(self, now: int) -> float:
        if not hasattr(self, '_ot'):
            import numpy as np  # research only
            z = np.load(self.p['s_oracle'])
            self._ot, self._os, self._oi = z['t'].tolist(), z['S'].tolist(), 0
        t, i = self._ot, self._oi
        while i < len(t) and t[i] <= now:
            i += 1
        self._oi = i
        return self._os[i - 1] if i else 10000.0

    def _snipe(self, snapshot: OrderBookSnapshot, S: float) -> None:
        p = self.p
        for side, book, lim in ((BUY, snapshot.asks, math.floor(S - p['s_take'])),
                                (SELL, snapshot.bids, math.ceil(S + p['s_take']))):
            if self.ioc_live[side]:
                continue
            avail = 0
            for o in book:
                if o.owner != 'market':
                    continue
                if (side == BUY and o.price > lim) or (side == SELL and o.price < lim):
                    break
                avail += o.quantity
            if not avail:
                continue
            q = min(avail, p['s_take_max'], self._room(side))
            if q > 0 and lim > 0:
                self._send(side, lim, q, ioc=True)

    def _sig_guard(self, now: int, bb: int, qb: int, ba: int, qa: int) -> None:
        p = self.p
        f2 = 0
        if p['flow2_k']:
            lim2 = now - p['flow2_ms'] * 1000
            for ts, q in reversed(self.flow):
                if ts < lim2:
                    break
                f2 += q
        tot = qb + qa
        for side, q_side, against, against2, px in ((BUY, qb, -self.flow_sum, -f2, bb),
                                                   (SELL, qa, self.flow_sum, f2, ba)):
            pk_px, pk = self.peak[side]
            pk = max(pk, q_side) if pk_px == px else q_side
            self.peak[side] = (px, pk)
            if ba - bb > 1:
                self.unsafe[side] = False
                continue
            score = q_side / tot - p['flow_k'] * against / 1000.0 - p['flow2_k'] * against2 / 1000.0
            bad = score < p['g_share'] and not (p['g_dep'] and q_side >= p['g_dep'] * pk)
            ok = score >= p['g_share'] or (p['g_dep'] and q_side >= p['g_dep'] * pk)
            if bad:
                self.unsafe[side] = True
            elif ok:
                self.unsafe[side] = False
