"""r2_gen: r2_ladder plus mechanics taken from the reverse-engineered round 2 generator
(marek/research/round2/round2_generator.md). With s_exp_vol = s_gap_vol = 0 and vt_on = js_on = 0 it is
r2_ladder exactly. New mechanics:
    s_exp_vol  impact exponent for S while volatile (generator: c*filled^0.61 vol, ^0.625 calm); 0 = s_exp.
    s_gap_vol  while volatile, every new touch insert bounds S by this gap (generator: no volatile quote is
               ever within 0.70 of S); 0 = s_gap.
    vt_on      fast regime tell: share of the last vt_n new touch inserts that land within vt_near of S.
               Volatile below vt_in, calm above vt_out. vt_mode 1: the tell alone decides; 2: volatile if
               the tell or the spread EMA says so.
    js_on      post-jump sweep. From js_w0 to js_w1 s after the drought is detected, scan new inserts: a bid
               more than js_jmin below the pre-jump S (or an ask more than js_clip below it) votes "down", and
               the mirror votes "up". With js_votes agreeing votes, IOC into the stale side: down -> sell every
               bid >= ceil(max new far bid + js_dhat + js_m), or only the best bid with clip evidence alone.
               S is re-anchored to the far insert + js_dhat.

--- r2_ladder docstring ---
r2_ladder: s_ladder plus the round 2 mechanics (ROUND2_STRAT.md). Every new mechanic sits behind a
PARAMS flag; with sg_on = 0 and vol_on = 0 this is s_ladder exactly (-194,451 on round2). Defaults are the
tuned round 2 config (round2 +365k; mean 428k over 6 synthetic tapes + round2).

Round 2 = round 1 generator + a calm/volatile regime + 27 scheduled shocks (3.0 s maker drought, then an
exogenous jump of S ~ N(0, 43) ticks). New mechanics:
    sg_on     shock guard: count market inserts (book grew by one order) in a sg_win_ms window; below sg_thr
              -> drought. Cancel every resting order and stay out until sg_hold_s after detection.
    sg_flat   during the guard, also flatten inventory: IOC at the touch (1) or not (0).
    vol_on    volatile-regime detector: time-weighted spread EMA (tau vol_tau_s) > vol_in enters, < vol_out
              for vol_exit_s leaves.
    vol_mode  0 = ignore, 1 = flat (no adding quotes, keep flattening at the touch), 2 = wide quotes at
              >= vol_dist from S with cap vol_cap, 3 = flat + snipe stale quotes through S by vol_take,
              4 = wide quotes + snipe.
    s_c_vol   impact coefficient for S while the volatile regime is detected (0 = use s_c).

--- s_ladder docstring ---
s_ladder: sig_ladder's queue-priority ladder driven by the hidden fair price S (round1 research).

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
    'topup': 30,
    'trade_start_us': 0,
    # hidden price S
    's_c': 0.026,
    's_exp': 0.5,
    's_gap': 0.095,  # makers never quote within this of S
    # S-driven quoting
    's_pull': 0.33,  # 0 = off
    's_hyst': 0.035,
    's_pull_flat': 0.13,  # the flattening part is pulled too once dist < this (-9 = never)
    's_pull_pos': -0.03,  # adding-side threshold = s_pull + s_pull_pos * |pos| / cap
    's_keep_flat': 1,  # while bad keep min(resting, |pos|) at the touch on the side that flattens
    's_prejoin': 1,  # while bad rest the freed room one tick behind the touch
    's_gen': 1,  # 1: quote each side at the best price p with dist(p) >= threshold (the pre-joined order follows S)
    's_maxdepth': 2,
    's_deep': 0,  # s_gen: while good also rest this many lots one tick behind the touch (room after the touch order)
    's_qk': 0.0,  # s_gen: threshold += s_qk * s_c * sqrt(market lots queued ahead of our touch order, or the whole touch)  # s_gen: don't rest more than this many ticks behind the touch
    's_take': 0.035,  # 0 = off; IOC when the far touch is at least this far through S
    's_take_max': 100,
    's_improve': 0.2,  # 0 = off
    # sig_ladder guard (defaults = sig_ladder's)
    'use_sig': 0,
    'g_share': 0.10,
    'g_dep': 0.5,
    'flow_k': 0.10,
    'flow_ms': 1000,
    'flow2_k': 0.5,
    'flow2_ms': 200,
    'improve_both': 1,
    # ---- round 2
    'debug': 0,
    # shock guard
    'sg_on': 1,
    'sg_win_ms': 500,
    'sg_thr': 50,
    'sg_hold_s': 3.5,
    'sg_cool_s': 10.0,
    'sg_flat': 2,  # 0 = cancel only, 1 = + IOC flatten at the touch, 2 = keep flattening passively at the touch
    # volatile regime
    'vol_on': 1,
    'vol_tau_s': 1.0,
    'vol_in': 1.5,
    'vol_out': 1.2,
    'vol_exit_s': 3.0,
    'vol_mode': 4,
    'vol_dist': 2.5,
    'vol_size': 50,
    'vol_cap': 100,
    'vol_take': 1.5,
    's_c_vol': 0.20,
    # ---- r2_gen
    's_exp_vol': 0.0,
    's_gap_vol': 0.0,
    'vt_on': 0,
    'vt_mode': 1,
    'vt_n': 10,
    'vt_near': 0.7,
    'vt_in': 0.3,
    'vt_out': 0.6,
    'js_on': 0,
    'js_w0': 2.40,
    'js_w1': 3.40,
    'js_jmin': 4.0,
    'js_clip': 1.0,
    'js_votes': 2,
    'js_dhat': 0.5,
    'js_m': 1.0,
    'js_cap': 100,
}

PARAMS.update({'js_on': 1, 'vol_size': 100, 'vol_take': 1.2, 'vol_dist': 2.2})  # GA g1_b
BUY, SELL = Side.BUY, Side.SELL


class R2Gen(BaseStrategy):
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
        # round 2 state
        self.S = None
        self.prev_n = None
        self.ins: deque = deque()
        self.t_first = None
        self.sg_until = -1
        self.sg_cool = -1
        self.sg_n = 0
        self.vol = False
        self.vol_ema = 1.0
        self.vol_last_t = None
        self.vol_last_spr = 1
        self.vol_low_since = None
        # r2_gen state
        self.vt: deque = deque()
        self.vt_sum = 0
        self.vt_vol = False
        self.ema_vol = False
        self.sg_t0 = -1
        self.js = None  # dict while a shock window is being watched

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

    def _room(self, side: Side, cap: int | None = None) -> int:
        cap = self.p['cap'] if cap is None else cap
        rest = sum(o[2] for o in self.orders.values() if o[0] == side)
        if side == BUY:
            return cap - (self.pos + rest)
        return cap + self.pos - rest

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
                c = p['s_c_vol'] if (self.vol and p['s_c_vol']) else p['s_c']
                e = p['s_exp_vol'] if (self.vol and p['s_exp_vol']) else p['s_exp']
                self.s_raw += sg * c * q ** e

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
        p = self.p
        if p['vt_on']:
            d = (s - px) if side == BUY else (px - s)
            near = 1 if d < p['vt_near'] else 0
            self.vt.append(near)
            self.vt_sum += near
            if len(self.vt) > p['vt_n']:
                self.vt_sum -= self.vt.popleft()
            if len(self.vt) >= p['vt_n']:
                sh = self.vt_sum / len(self.vt)
                if sh < p['vt_in']:
                    self.vt_vol = True
                elif sh > p['vt_out']:
                    self.vt_vol = False
        g = p['s_gap_vol'] if (self.vol and p['s_gap_vol']) else p['s_gap']
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
        S = self._s_value(now)
        self.S = S

        # ---- round 2 detectors
        if p['sg_on'] and self._shock_check(now, snapshot):
            if p['js_on'] and self._jump_sweep(now, snapshot, S, bb, ba):
                return
            if p['sg_flat'] == 2:
                self._vol_act(snapshot, S, bb, ba, 1)
            else:
                self._guard(bb, ba)
            return
        if p['vol_on']:
            self._vol_update(now, ba - bb)
            if self.vol and p['vol_mode']:
                self._vol_act(snapshot, S, bb, ba, p['vol_mode'])
                return

        if ba - bb > p['max_spread'] or now < p['trade_start_us']:
            return

        # ---- snipe stale quotes through S
        if p['s_take']:
            self._snipe(snapshot, S, p['s_take'], None)

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
            self._quote_gen(S, bb, ba, qb_px, qa_px, snapshot)
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

    @staticmethod
    def _ahead(levels, price: int) -> int:
        """Market lots ahead of our first order at `price`, or the whole level if we have none there."""
        q = 0
        for o in levels:
            if o.price != price:
                if q:
                    break
                continue
            if o.owner != 'market':
                return q
            q += o.quantity
        return q

    def _quote_gen(self, S: float, bb: int, ba: int, qb_px: int, qa_px: int, snapshot) -> None:
        p = self.p
        for side, base, mt, step in ((BUY, qb_px, bb, -1), (SELL, qa_px, ba, 1)):
            flat = (side == SELL and self.pos > 0) or (side == BUY and self.pos < 0)
            th = p['s_pull'] if flat else p['s_pull'] + p['s_pull_pos'] * abs(self.pos) / p['cap']
            if p['s_qk'] and not flat:
                th += p['s_qk'] * p['s_c'] * self._ahead(snapshot.bids if side == BUY else snapshot.asks, base) ** 0.5
            if side == BUY:
                tgt = base if S - base >= th else min(base - 1, math.floor(S - th))
            else:
                tgt = base if base - S >= th else max(base + 1, math.ceil(S + th))
            good = tgt == base
            mine = sorted(((o[3], oid, o) for oid, o in self.orders.items()
                           if o[0] == side and not o[4] and oid not in self.cancelling), key=lambda x: x[0])
            at, ahead, deep = [], [], []
            for _, oid, o in mine:
                d = (o[1] - tgt) if side == BUY else (tgt - o[1])
                if d == 0:
                    at.append((oid, o))
                elif d == -1 and good and p['s_deep']:
                    deep.append((oid, o))
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
            if good and p['s_deep'] and tgt + step > 0:
                have = sum(o[2] for _, o in deep)
                target = p['s_deep']
                if have:
                    short = target - have
                    target = short if p['topup'] and short >= p['topup'] else 0
                q = min(target, self._room(side))
                if q > 0:
                    self._send(side, tgt + step, q)

    # ------------------------------------------------------------------ round 2
    def _shock_check(self, now: int, snapshot: OrderBookSnapshot) -> bool:
        """True while the shock guard is active (drought detected, jump not yet absorbed)."""
        p = self.p
        n = len(snapshot.bids) + len(snapshot.asks)
        if self.prev_n is not None and n > self.prev_n:
            self.ins.append(now)
        self.prev_n = n
        if self.t_first is None:
            self.t_first = now
        win = p['sg_win_ms'] * 1000
        while self.ins and self.ins[0] < now - win:
            self.ins.popleft()
        if now < self.sg_until:
            return True
        if now - self.t_first < win or now < self.sg_cool:
            return False
        if len(self.ins) < p['sg_thr']:
            self.sg_n += 1
            self.sg_t0 = now
            self.js = None
            self.sg_until = now + int(p['sg_hold_s'] * 1e6)
            self.sg_cool = now + int(p['sg_cool_s'] * 1e6)
            if p['debug']:
                print(f'DBG shock {self.sg_n} t={now / 1e6:.3f} n={len(self.ins)} pos={self.pos}')
            return True
        return False

    def _jump_sweep(self, now: int, snapshot: OrderBookSnapshot, S: float, bb: int, ba: int) -> bool:
        """Post-jump sweep of the stale book. True when it handled this event (the guard is skipped)."""
        p = self.p
        dt = now - self.sg_t0
        if dt < p['js_w0'] * 1e6 or dt > p['js_w1'] * 1e6:
            return False
        js = self.js
        if js is None:
            mx = 0
            for o in snapshot.bids:
                if o.owner == 'market':
                    mx = max(mx, int(o.order_id))
            for o in snapshot.asks:
                if o.owner == 'market':
                    mx = max(mx, int(o.order_id))
            self.js = {'S0': S, 'mx': mx, 'up': 0, 'dn': 0, 'fb': None, 'fa': None, 'dir': 0}
            return False
        S0, mx = js['S0'], js['mx']
        for o in snapshot.bids:
            if o.owner != 'market':
                continue
            i = int(o.order_id)
            if i <= js['mx']:
                continue
            mx = max(mx, i)
            if o.price < S0 - p['js_jmin']:
                js['dn'] += 1
                js['fb'] = o.price if js['fb'] is None else max(js['fb'], o.price)
            elif o.price > S0 + p['js_clip']:
                js['up'] += 1
        for o in snapshot.asks:
            if o.owner != 'market':
                continue
            i = int(o.order_id)
            if i <= js['mx']:
                continue
            mx = max(mx, i)
            if o.price > S0 + p['js_jmin']:
                js['up'] += 1
                js['fa'] = o.price if js['fa'] is None else min(js['fa'], o.price)
            elif o.price < S0 - p['js_clip']:
                js['dn'] += 1
        js['mx'] = mx
        if not js['dir']:
            if js['dn'] >= p['js_votes'] and js['dn'] >= 2 * js['up']:
                js['dir'] = -1
            elif js['up'] >= p['js_votes'] and js['up'] >= 2 * js['dn']:
                js['dir'] = 1
            else:
                return False
            if p['debug']:
                print(f'DBG jump dir={js["dir"]} t={now / 1e6:.3f} dt={dt / 1e6:.3f} S0={S0:.1f} '
                      f'fb={js["fb"]} fa={js["fa"]} bb={bb} ba={ba} pos={self.pos}')
        # direction known: nothing resting, re-anchor S, sweep the stale side
        for oid, o in list(self.orders.items()):
            if not o[4] and oid not in self.cancelling:
                self._cancel(oid)
        cur = self.s_raw + self.off
        if js['dir'] < 0:
            if js['fb'] is not None:
                s_new = js['fb'] + p['js_dhat']
                if cur > s_new:
                    self.off += s_new - cur
                lim = math.ceil(s_new + p['js_m'])
            else:
                lim = bb
            if not self.ioc_live[SELL] and lim > 0:
                avail = sum(o.quantity for o in snapshot.bids if o.owner == 'market' and o.price >= lim)
                q = min(avail, self._room(SELL, p['js_cap']))
                if q > 0:
                    self._send(SELL, lim, q, ioc=True)
        else:
            if js['fa'] is not None:
                s_new = js['fa'] - p['js_dhat']
                if cur < s_new:
                    self.off += s_new - cur
                lim = math.floor(s_new - p['js_m'])
            else:
                lim = ba
            if not self.ioc_live[BUY] and lim > 0:
                avail = sum(o.quantity for o in snapshot.asks if o.owner == 'market' and o.price <= lim)
                q = min(avail, self._room(BUY, p['js_cap']))
                if q > 0:
                    self._send(BUY, lim, q, ioc=True)
        return True

    def _guard(self, bb: int, ba: int) -> None:
        """Shock guard: nothing resting; optionally flatten with an IOC at the touch."""
        for oid, o in list(self.orders.items()):
            if not o[4] and oid not in self.cancelling:
                self._cancel(oid)
        if self.p['sg_flat'] and self.pos and not (self.ioc_live[BUY] or self.ioc_live[SELL]):
            if self.pos > 0:
                self._send(SELL, bb, self.pos, ioc=True)
            else:
                self._send(BUY, ba, -self.pos, ioc=True)

    def _vol_update(self, now: int, spread: int) -> None:
        p = self.p
        if self.vol_last_t is not None:
            dt = (now - self.vol_last_t) / 1e6
            a = math.exp(-dt / p['vol_tau_s'])
            self.vol_ema = a * self.vol_ema + (1 - a) * self.vol_last_spr
        self.vol_last_t = now
        self.vol_last_spr = spread
        if not self.ema_vol:
            if self.vol_ema > p['vol_in']:
                self.ema_vol = True
                self.vol_low_since = None
        else:
            if self.vol_ema < p['vol_out']:
                if self.vol_low_since is None:
                    self.vol_low_since = now
                elif now - self.vol_low_since >= p['vol_exit_s'] * 1e6:
                    self.ema_vol = False
            else:
                self.vol_low_since = None
        was = self.vol
        if not p['vt_on']:
            self.vol = self.ema_vol
        elif p['vt_mode'] == 1:
            self.vol = self.vt_vol
        else:
            self.vol = self.vt_vol or self.ema_vol
        if p['debug'] and was != self.vol:
            print(f'DBG vol={int(self.vol)} t={now / 1e6:.3f} pos={self.pos}')

    def _vol_act(self, snapshot: OrderBookSnapshot, S: float, bb: int, ba: int, mode: int) -> None:
        """Volatile regime. 1: flatten only at the touch. 2: wide quotes. 3: flatten + snipe."""
        p = self.p
        if mode >= 3:
            self._snipe(snapshot, S, p['vol_take'], p['vol_cap'])
            mode = 2 if mode == 4 else 1
        for side, touch in ((BUY, bb), (SELL, ba)):
            flat_q = -self.pos if side == BUY else self.pos
            if mode == 2:
                if side == BUY:
                    px = min(math.floor(S - p['vol_dist']), ba - 1)
                else:
                    px = max(math.ceil(S + p['vol_dist']), bb + 1)
                want = px
                lim = p['vol_cap']
                qty = p['vol_size']
            else:
                want = touch
                lim = 0
                qty = 0
            keep = max(flat_q, 0)
            have = 0
            for oid, o in list(self.orders.items()):
                if o[0] != side or o[4] or oid in self.cancelling:
                    continue
                if o[1] != want:
                    self._cancel(oid)
                    continue
                have += min(o[2], self.modifying.get(oid, o[2]))
            if mode == 2:
                room = self._room(side, lim)
                tgt = max(qty, keep)
                q = min(tgt - have, room) if have < tgt else 0
                if q > 0 and (have == 0 or q >= p['topup']) and want > 0:
                    self._send(side, want, q)
            else:
                # flatten-only: shrink anything above |pos|, add the missing flattening size
                if have > keep:
                    k = keep
                    for oid, o in list(self.orders.items()):
                        if o[0] == side and not o[4] and oid not in self.cancelling and o[1] == want:
                            k = self._shrink(oid, o, k)
                elif keep - have >= max(1, p['topup'] if have else 1) and want > 0:
                    self._send(side, want, keep - have)

    def _s_value(self, now: int) -> float:
        """Current S estimate (research subclasses override this, e.g. with the tape's exact S)."""
        return self.s_raw + self.off

    def _snipe(self, snapshot: OrderBookSnapshot, S: float, take: float, cap: int | None) -> None:
        p = self.p
        for side, book, lim in ((BUY, snapshot.asks, math.floor(S - take)),
                                (SELL, snapshot.bids, math.ceil(S + take))):
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
            q = min(avail, p['s_take_max'], self._room(side, cap))
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
