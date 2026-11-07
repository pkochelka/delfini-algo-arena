"""r2_mm: fair-price market maker for the round-2 generator (round 1 + exogenous jumps in turbulent regimes).

Fair price.  Round 2 still looks like the round-1 generator (makers quote floor(S - D) / ceil(S + D), D >= 0.05,
clipped so they never cross; takers move S by c * sqrt(filled)), but S also jumps by itself. So S is tracked as
S = U + off, U = c * sum(sign * sqrt(filled)) over market takers, and off is bracketed by [lo, hi]:
    every maker bid insert X:  off >= X + gap - U      every maker ask insert X:  off <= X - gap - U
(valid whether or not the insert was clipped). The bracket widens by `relax` ticks/s (small extra moves). An insert
that falls outside the bracket means S moved: the bracket is reset to [v, v + reset_w] (bid) / [v - reset_w, v]
(ask). A reset larger than `jump_th` is a JUMP.

Each round-2 finding is a hypothesis behind a flag (round2_findings.md; results in r2_hypotheses.md):
  H1 fair      quote around the bracket S, not the touch (calm periods look like round 1)
  H2 take      IOC stale quotes beyond the hard bracket bound (after a jump the stale side is many ticks off)
  H3 jump_*    on a jump: cancel the stale side at once and pause passive quoting for jump_pause s
  H4 turb_*    jumps cluster (P(next within 10 s | jump) = 0.7): after a jump / under high realised vol,
               shrink the passive cap to turb_cap and optionally quote only the inventory-reducing side
  H5 big_*     fills against >100-lot takers mark out negative: sit only at one price level (keep_depth 0)
  H6 sig_*     vol clustering: widen theta by vol_k * (sigma_fast / sigma_ref - 1)
  H7 depth_*   touch depth falls in turbulence: cap <= depth_k * EWMA(touch qty)
  H8 cburst_*  one-sided cancel bursts precede the visible reset (cancel hazard rises with adverse move):
               cancel our quotes on the side whose market queue is being cancelled away
  H9 ks_*      equity kill switch (robust_ladder): drawdown -> defensive cap with escalating cooldown
  H10 inv_skew lean quotes against inventory (keeps room for the next snipe)
  H11 w_k      quote a wide bracket with theta widened by its width
  H12 pcap     lower cap for passive quotes, leaving room for takes
  H13 imb_k    imbalance tilt of S        H14 take_red  looser inventory-reducing takes
  H15 empty_*  quote into an empty side   H16 turb_spread  market spread >= k counts as turbulent
  H17 turb_tk  taker-rate trigger         H18 c_turb  impact multiplier in turbulence
  H19 mom_*    momentum tilt of S
Defaults = the kept ones (H1, H2, H4 + turb_off, H11, H16): 12 r2 folds mean +16.7k, min +6.4k; holdout +27.6k;
full round2_public +228k, no breach. Dominated by fv_mm.py (+36.4k / fold, holdout +122k); see r2_hypotheses.md.
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
    'trade_start_us': 0,
    'log': 0,  # record fills in self.fills (diagnostics only)
    'cap': 100,
    # fair price
    'c': 0.037,
    'gap': 0.05,
    'relax': 0.0,  # ticks/s the bracket widens by on each side
    'reset_w': 1.0,  # width of the bracket after a reset
    'max_width': 1.2,  # quote only while hi - lo <= max_width
    'unsure_cancel': 1,  # cancel resting quotes while the bracket is too wide
    'w0': 0.3,  # H11: theta += w_k * max(0, (hi - lo) - w0): quote a wide bracket with a wider margin
    'w_k': 1.0,
    'imb_k': 0.0,  # H13: S += imb_k * (qb - qa) / (qb + qa) (market touch queues)
    'c_turb': 1.0,  # H18: impact multiplier while turbulent
    'mom_k': 0.0,  # H19: S += mom_k * (S - S mom_s seconds ago), 100 ms samples
    'mom_s': 10.0,
    'pcap': 100,  # H12: passive quotes keep the worst case within pcap (takes may use the full cap)
    # H1 quoting (s_mm core)
    'theta': 0.30,
    'theta_keep': 0.10,
    'keep_depth': 1,
    'reach_off': 1.0,
    'size_top': 1000,
    'min_post': 1,
    # H2 taking
    'take': 1,
    'take_lb': 0.0,  # buy when ask <= (U + lo) - take_lb; sell when bid >= (U + hi) + take_lb
    'take_mid': 0.15,  # also: buy when ask <= S - take_mid while the bracket is tight (s_mm rule), 0 = off
    'take_max': 100,
    'take_red': 0.0,  # H14: inventory-reducing takes get this much looser limits (scaled by |pos| / cap)
    'take_red_turb': 0,  # 1: only while turbulent
    # H3 jumps
    'jump_th': 2.0,
    'jump_cancel': 1,
    'jump_pause': 0.0,  # s without passive quotes after a jump
    # H4 turbulence
    'turb_hold': 0.0,  # s after a jump counted as turbulent (0 = off)
    'turb_rv': 2.0,  # realised vol of S (ticks/sqrt s, 10 s EWMA) above which it is turbulent (0 = off)
    'turb_cap': 0,
    'turb_reduce': 0,  # turbulent: passive quotes only on the side that reduces |pos|
    'turb_theta': 0.0,  # extra theta while turbulent
    'turb_spread': 3,  # H16: market spread >= this counts as turbulent (0 = off)
    'turb_spread_hold': 0.0,  # s a wide spread keeps the turbulent flag on
    'turb_tk': 0.0,  # H17: market taker rate (EWMA, 1 s half-life, /s) above this is turbulent (0 = off)
    'empty_quote': 0,  # H15: quote into an empty side (the missing touch is taken as S +- empty_off)
    'empty_off': 3,
    'turb_off': 1,  # turbulent: no passive quotes at all (taking only)
    # H6 volatility-scaled theta
    'vol_k': 0.0,
    'sig_ref': 1.0,
    # H7 depth-scaled cap
    'depth_k': 0.0,
    'depth_min': 25,
    'depth_hl': 10.0,  # s
    # H8 cancel bursts
    'cburst_n': 0,  # >0: this many market cancels at our touch side within cburst_ms -> pull that side
    'cburst_ms': 30.0,
    'cburst_pause': 0.05,
    # H9 kill switch
    'ks_dd': 0.0,  # $ drop below the trailing ks_w1 peak (0 = off)
    'ks_w1': 60,
    'ks_cap': 20,
    'ks_cool': 120.0,
    'ks_mult': 2.0,
    'ks_cool_max': 1800.0,
    'ks_reset': 900.0,
    # H10
    'inv_skew': 0.0,  # theta += inv_skew * pos / cap on the adding side, -= on the reducing side
}

BUY, SELL = Side.BUY, Side.SELL


class R2MM(BaseStrategy):
    def __init__(self) -> None:
        super().__init__()
        self.p = PARAMS
        p = self.p
        self.pos = 0
        self.cash = 0.0
        self.orders: dict[str, list] = {}  # id -> [side, price, remaining]
        self.cancelling: set[str] = set()
        # fair price
        self.U = 0.0
        self.lo = -1e18
        self.hi = 1e18
        self.t_relax = 0
        self.max_id = ''
        self.S = None
        # regime
        self.jumps = 0
        self.last_jump_us = -10**15
        self.last_jump_sign = 0
        self.next_rv_us = 0
        self.rv_S = None
        self.var_f = (p['sig_ref'] ** 2)  # fast EWMA of squared 1 s-normalised S changes
        self.depth = -1.0
        self.next_depth_us = 0
        # cancel bursts: market order ids at our touch level, per side
        self.touch_ids = {BUY: set(), SELL: set()}
        self.touch_px = {BUY: None, SELL: None}
        self.cxl = {BUY: deque(), SELL: deque()}
        self.pull_until = {BUY: -1, SELL: -1}
        # kill switch
        self.eq: deque = deque()
        self.next_eq_us = 0
        self.def_until = -1.0
        self.streak = 0
        self.def_cap = p['ks_cap']
        self.stats = {'jumps': 0, 'takes': 0, 'take_qty': 0, 'ks': 0, 'bursts': 0}
        self.take_ids: set[str] = set()
        self.tk_rate = 0.0
        self.tk_t = 0
        self.spread = 1
        self.wide_until = -1
        self.hist: deque = deque()
        self.fills: list = []  # p['log']: (t_us, +-1, price, qty, is_take, S, s since jump, turbulent)

    # ---------------------------------------------------------------- private feed
    def on_private_order_update(self, event: PrivateOrderEvent) -> None:
        o = self.orders.get(event.order_id)
        if o is None:
            return
        st = event.status
        if st is OrderStatus.PARTIALLY_FILLED or st is OrderStatus.FILLED:
            q = event.filled_quantity
            if o[0] is BUY:
                self.pos += q
                self.cash -= q * event.fill_price
            else:
                self.pos -= q
                self.cash += q * event.fill_price
            o[2] = event.remaining_quantity
            if self.p['log']:
                ts = event.timestamp_us
                self.fills.append((ts, 1 if o[0] is BUY else -1, event.fill_price, q, event.order_id in self.take_ids,
                                   self.S, (ts - self.last_jump_us) * 1e-6, self._turbulent(ts)))
        if st is OrderStatus.FILLED or st is OrderStatus.CANCELED or st is OrderStatus.REJECTED:
            self.orders.pop(event.order_id, None)
            self.cancelling.discard(event.order_id)
            self.take_ids.discard(event.order_id)

    def _newer(self, oid: str) -> bool:
        m = self.max_id
        return len(oid) > len(m) or (len(oid) == len(m) and oid > m)

    def _jump(self, ts: int, size: float) -> None:
        self.jumps += 1
        self.stats['jumps'] += 1
        self.last_jump_us = ts
        self.last_jump_sign = 1 if size > 0 else -1

    # ---------------------------------------------------------------- market feed
    def on_order_book_change(self, snapshot: OrderBookSnapshot, trades: list[Trade]) -> None:
        p = self.p
        ts = snapshot.timestamp_us
        if trades:
            q = 0
            sgn = 0
            tid = None
            for tr in trades:
                if tr.taker_owner == 'market':
                    q += tr.quantity
                    sgn = 1 if tr.book_side == 'ask' else -1
                    tid = tr.taker_order_id
            if q:
                if p['turb_tk']:
                    dt = (ts - self.tk_t) * 1e-6
                    self.tk_rate = self.tk_rate * 0.5 ** dt + math.log(2)  # 1 s half-life EWMA, in /s
                    self.tk_t = ts
                k = p['c_turb'] if p['c_turb'] != 1.0 and self._turbulent(ts) else 1.0
                self.U += sgn * k * p['c'] * math.sqrt(q)
                if self._newer(tid):
                    self.max_id = tid
        if p['relax'] and ts > self.t_relax:
            d = p['relax'] * (ts - self.t_relax) * 1e-6
            self.lo -= d
            self.hi += d
        self.t_relax = ts

        U = self.U
        gap = p['gap']
        jth = p['jump_th']
        rw = p['reset_w']
        bids, asks = snapshot.bids, snapshot.asks
        bb = ba = mbb = mba = None
        qb = qa = 0
        for o in bids:
            px = o.price
            if bb is None:
                bb = px
            if o.owner != 'market':
                continue
            if mbb is None:
                mbb = px
            if px == mbb:
                qb += o.quantity
            oid = o.order_id
            if self._newer(oid):
                self.max_id = oid
                v = px + gap - U
                if v > self.hi:
                    if v - self.hi > jth:
                        self._jump(ts, v - self.hi)
                    self.lo, self.hi = v, v + rw
                elif v > self.lo:
                    self.lo = v
        for o in asks:
            px = o.price
            if ba is None:
                ba = px
            if o.owner != 'market':
                continue
            if mba is None:
                mba = px
            if px == mba:
                qa += o.quantity
            oid = o.order_id
            if self._newer(oid):
                self.max_id = oid
                v = px - gap - U
                if v < self.lo:
                    if self.lo - v > jth:
                        self._jump(ts, v - self.lo)
                    self.lo, self.hi = v - rw, v
                elif v < self.hi:
                    self.hi = v
        if self.hi > 1e17 or self.lo < -1e17:
            return
        S = S_raw = U + 0.5 * (self.lo + self.hi)
        if p['mom_k'] and self.hist:
            S += p['mom_k'] * (S - self.hist[0])
        if p['imb_k'] and qb + qa:
            S += p['imb_k'] * (qb - qa) / (qb + qa)
        self.S = S
        if p['cburst_n']:
            self._bursts(ts, bids, asks, mbb, mba)
        self._regime(ts, S_raw, qb, qa, bb, ba)
        if ts < p['trade_start_us']:
            return
        if bb is None or ba is None:
            if not p['empty_quote'] or (bb is None and ba is None):
                return
            if bb is None:
                bb = min(ba - 1, math.floor(S - p['empty_off']))
            else:
                ba = max(bb + 1, math.ceil(S + p['empty_off']))
        self.spread = (mba - mbb) if (mba is not None and mbb is not None) else 99
        if p['turb_spread'] and self.spread >= p['turb_spread']:
            self.wide_until = ts + int(p['turb_spread_hold'] * 1e6) + 1
        self._take(ts, S, bb, ba, bids, asks)
        self._quote(ts, S, bb, ba)

    # ---------------------------------------------------------------- regime
    def _regime(self, ts: int, S: float, qb: int, qa: int, bb, ba) -> None:
        p = self.p
        if ts >= self.next_rv_us:  # 100 ms samples of S
            if self.rv_S is not None:
                d = S - self.rv_S
                a = 1 - 0.5 ** (0.1 / 10.0)  # 10 s half-life
                self.var_f += a * (d * d * 10 - self.var_f)  # per-second variance
            self.rv_S = S
            self.next_rv_us = ts + 100_000
            if p['mom_k']:
                self.hist.append(S)
                if len(self.hist) > p['mom_s'] * 10:
                    self.hist.popleft()
        if p['depth_k'] and ts >= self.next_depth_us and qb and qa:
            tq = 0.5 * (qb + qa)
            if self.depth < 0:
                self.depth = tq
            else:
                self.depth += (1 - 0.5 ** (0.1 / p['depth_hl'])) * (tq - self.depth)
            self.next_depth_us = ts + 100_000
        if p['ks_dd'] and ts >= self.next_eq_us and bb is not None and ba is not None:
            self._kill(ts * 1e-6, self.cash + self.pos * 0.5 * (bb + ba))
            self.next_eq_us = ts + 1_000_000

    def _kill(self, t_s: float, eq: float) -> None:
        p = self.p
        self.eq.append((t_s, eq))
        while self.eq and self.eq[0][0] < t_s - p['ks_w1']:
            self.eq.popleft()
        if t_s < self.def_until:
            return
        if self.streak and t_s - self.def_until > p['ks_reset']:
            self.streak = 0
        peak = max(e for _, e in self.eq)
        if eq < peak - p['ks_dd']:
            self.streak += 1
            self.stats['ks'] += 1
            self.def_until = t_s + min(p['ks_cool'] * p['ks_mult'] ** (self.streak - 1), p['ks_cool_max'])
            self.def_cap = int(p['ks_cap'] * 0.5 ** (self.streak - 1))
            self.eq.clear()
            self.eq.append((t_s, eq))

    def _turbulent(self, ts: int) -> bool:
        p = self.p
        if p['turb_spread'] and ts < self.wide_until:
            return True
        if p['turb_tk'] and self.tk_rate * 0.5 ** ((ts - self.tk_t) * 1e-6) > p['turb_tk']:
            return True
        if p['turb_hold'] and ts - self.last_jump_us < p['turb_hold'] * 1e6:
            return True
        return bool(p['turb_rv']) and self.var_f > p['turb_rv'] ** 2

    def _bursts(self, ts: int, bids, asks, mbb, mba) -> None:
        """Count market cancels at the market touch (orders that vanish without a trade at the same level)."""
        p = self.p
        win = p['cburst_ms'] * 1000
        for side, views, px in ((BUY, bids, mbb), (SELL, asks, mba)):
            ids = set()
            for o in views:
                if o.price != px:
                    break
                if o.owner == 'market':
                    ids.add(o.order_id)
            if px == self.touch_px[side]:
                gone = len(self.touch_ids[side] - ids)
                dq = self.cxl[side]
                for _ in range(gone):
                    dq.append(ts)
                while dq and dq[0] < ts - win:
                    dq.popleft()
                if len(dq) >= p['cburst_n'] and ts >= self.pull_until[side]:
                    self.pull_until[side] = ts + int(p['cburst_pause'] * 1e6)
                    self.stats['bursts'] += 1
            self.touch_ids[side] = ids
            self.touch_px[side] = px

    # ---------------------------------------------------------------- taking
    def _rest(self, side: Side) -> int:
        return sum(o[2] for o in self.orders.values() if o[0] is side)

    def _take(self, ts: int, S: float, bb: int, ba: int, bids, asks) -> None:
        """IOC every market quote beyond the limit: buy asks <= lim_b, sell bids >= lim_a."""
        p = self.p
        if not p['take']:
            return
        U = self.U
        lim_b = U + self.lo - p['take_lb']
        lim_a = U + self.hi + p['take_lb']
        if p['take_mid'] and self.hi - self.lo <= p['max_width']:
            lim_b = max(lim_b, S - p['take_mid'])
            lim_a = min(lim_a, S + p['take_mid'])
        pos = self.pos
        if p['take_red'] and pos and (not p['take_red_turb'] or self._turbulent(ts)):
            r = p['take_red'] * abs(pos) / p['cap']
            if pos < 0:
                lim_b += r
            else:
                lim_a -= r
        if ba <= lim_b:
            self._ioc(BUY, math.floor(lim_b), asks)
        if bb >= lim_a:
            self._ioc(SELL, math.ceil(lim_a), bids)

    def _ioc(self, side: Side, lim: int, views) -> None:
        buy = side is BUY
        q = 0
        for o in views:
            if (o.price > lim) if buy else (o.price < lim):
                break
            if o.owner == 'market':
                q += o.quantity
        room = self.p['cap'] - self.pos if buy else self.p['cap'] + self.pos
        q = min(q, room - self._rest(side), self.p['take_max'])
        if q > 0:
            oid = self.send_order(OrderRequest(side=side, price=lim, quantity=q, order_type=OrderType.IOC))
            self.orders[oid] = [side, lim, q]
            self.cancelling.add(oid)
            self.take_ids.add(oid)
            self.stats['takes'] += 1
            self.stats['take_qty'] += q

    # ---------------------------------------------------------------- quoting
    def _cancel_side(self, side: Side) -> None:
        for oid, o in self.orders.items():
            if o[0] is side and oid not in self.cancelling:
                self.cancelling.add(oid)
                self.cancel_order(oid)

    def _quote(self, ts: int, S: float, bb: int, ba: int) -> None:
        p = self.p
        cap = p['cap']
        pos = self.pos
        since_jump = ts - self.last_jump_us
        if p['jump_cancel'] and since_jump == 0:
            self._cancel_side(SELL if self.last_jump_sign > 0 else BUY)
        if p['log']:
            w = self.hi - self.lo
            k = 'wide' if w > p['max_width'] else ('turb' if self._turbulent(ts) else 'quote')
            self.stats[k] = self.stats.get(k, 0) + 1
            self.stats['w_sum'] = self.stats.get('w_sum', 0) + min(w, 5)
            self.stats['pos_abs'] = self.stats.get('pos_abs', 0) + abs(self.pos)
        if self.hi - self.lo > p['max_width'] or since_jump < p['jump_pause'] * 1e6:
            if p['unsure_cancel']:
                self._cancel_side(BUY)
                self._cancel_side(SELL)
            return
        # passive cap for this event
        pc = min(cap, p['pcap'])
        turb = self._turbulent(ts)
        if turb:
            pc = min(pc, p['turb_cap'])
        if p['ks_dd'] and ts * 1e-6 < self.def_until:
            pc = min(pc, self.def_cap)
        if p['depth_k'] and self.depth >= 0:
            pc = min(pc, max(p['depth_min'], int(p['depth_k'] * self.depth)))
        th = p['theta'] + (p['turb_theta'] if turb else 0.0)
        if p['w_k']:
            th += p['w_k'] * max(0.0, self.hi - self.lo - p['w0'])
        if p['vol_k']:
            th += p['vol_k'] * max(0.0, math.sqrt(self.var_f) / p['sig_ref'] - 1)
        sk = p['inv_skew'] * pos / cap
        quote_b = quote_a = True
        if turb and p['turb_reduce']:
            quote_b, quote_a = pos < 0, pos > 0
        if turb and p['turb_off']:
            quote_b = quote_a = False
        if ts < self.pull_until[BUY]:
            quote_b = False
        if ts < self.pull_until[SELL]:
            quote_a = False
        tb = min(math.floor(S - th - sk), ba - 1)
        ta = max(math.ceil(S + th - sk), bb + 1)
        tkb = p['theta_keep'] + sk
        tka = p['theta_keep'] - sk
        kd = p['keep_depth']
        rb = round(S - p['reach_off'])
        ra = round(S + p['reach_off'])
        rest_b = rest_a = 0
        have_b = have_a = False
        for oid, o in self.orders.items():
            if oid in self.cancelling:
                if o[0] is BUY:
                    rest_b += o[2]
                else:
                    rest_a += o[2]
                continue
            px = o[1]
            if o[0] is BUY:
                ok = quote_b and S - px >= tkb and tb - kd <= px < ba and px >= rb
                rest_b += o[2]
                if ok:
                    have_b = have_b or px == tb
                else:
                    self.cancelling.add(oid)
                    self.cancel_order(oid)
            else:
                ok = quote_a and px - S >= tka and bb < px <= ta + kd and px <= ra
                rest_a += o[2]
                if ok:
                    have_a = have_a or px == ta
                else:
                    self.cancelling.add(oid)
                    self.cancel_order(oid)
        mp = p['min_post']
        st = p['size_top']
        # worst case after every passive order on a side fills stays within the passive cap pc (<= cap)
        if quote_b and not have_b:
            q = min(pc - pos - rest_b, st)
            if q >= mp:
                oid = self.send_order(OrderRequest(side=BUY, price=tb, quantity=q, order_type=OrderType.LIMIT))
                self.orders[oid] = [BUY, tb, q]
        if quote_a and not have_a:
            q = min(pc + pos - rest_a, st)
            if q >= mp:
                oid = self.send_order(OrderRequest(side=SELL, price=ta, quantity=q, order_type=OrderType.LIMIT))
                self.orders[oid] = [SELL, ta, q]
