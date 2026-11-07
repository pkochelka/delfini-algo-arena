"""fv_mm: market maker + stale-quote taker around a Bayesian estimate of the generator's hidden price S.

Round 2 broke s_mm: S still drives the makers (bid = min(floor(S - D), ask - 1), ask mirrored, D = d0 + Exp(lam))
and the takers' impact (c * sign * sqrt(filled)), but S now also diffuses and jumps (194 jumps > 5 ticks,
clustered into turbulent regimes, round2_findings.md), so the exact bound tracker of s_mm keeps contradicting
itself. Here S is a posterior on a grid (class FV):
    predict   diffusion sigma + jumps (flat over the grid) at a self-exciting rate rho0 + exc
    makers    likelihood of each new market insert under the maker rule, censored when clamped at the
              opposite touch, plus an outlier share eta; two agreeing inserts far outside the grid restart it
    takers    impact shifts the grid (filled qty from the trades of one market event)
The posterior mean F is the fair value; its sd the uncertainty. After a jump the new level shows up in the
first one or two maker inserts on the far side, while the stale side of the book is only swept by takers
tens of ms later: the take rule (ask <= F - take) snipes it, and the keep rule cancels our own stale quotes.

Quoting (s_mm's structure, thresholds widened by sd and by turbulence):
    post   bid floor(F - th), ask ceil(F + th), th = theta + theta_sd * sd, never crossing
    keep   a resting order while its edge >= theta_keep (+ the same widening), reachable by the next taker,
           at most keep_depth ticks behind target
    take   IOC every market level priced at least `take + take_sd * sd` through F, up to the room
    room   pos + every order on the side (incl. cancels and IOCs in flight) <= cap, so no breach is possible
Offline parameter fit of the filter: r2_fv_fit.py.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

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

FV_PARAMS = {
    'h': 0.05,  # grid cell (ticks), 1/integer
    'K': 240,  # grid half-width in cells
    'regimes': 2,  # 2: joint calm/volatile HMM (round-2 generator); 1: calm laws only (the first version)
    # calm laws (marek/research/round2/s_dynamics/params_v3.json): D = d0 + Exp(lam), dS = c * q^e + N(0, jv)
    'd0': 0.049,
    'lam': 0.45,
    'c': 0.0142,
    'e': 0.625,
    'jv': 0.0155,
    # volatile laws, plus residual Laplace(bj_v) jumps at lamj_v / s
    'd0_v': 0.699,
    'lam_v': 0.907,
    'c_v': 0.0741,
    'e_v': 0.606,
    'jv_v': 0.076,
    'lamj_v': 0.23,
    'bj_v': 0.84,
    'q_cv': 0.004,  # regime switch rates (/s)
    'q_vc': 0.023,
    # maker law during a shock drought (both regimes)
    'd0_d': 0.54,
    'lam_d': 1.33,
    'eta': 0.005,  # outlier share of maker inserts
    'sigma': 0.3,  # extra diffusion, ticks / sqrt(s) (the generator has none; insurance against tapes that do)
    'rho0': 0.05,  # base rate of jumps flat over the grid (/s): robustness to unmodelled jumps
    'rho_shock': 30.0,  # the same inside an expected-shock window
    'kick': 1.0,  # excitation added per detected jump (/s)
    'exc_hl': 1.5,  # s
    'exc_max': 3.0,
    'jump_sd': 0.5,  # a jump is detected only by a confident filter (sd below this) ...
    'jump_det': 2.0,  # ... whose mean one observation moves this far (ticks)
    'w_u': 0.6,  # taker-price model (offline only: the live feed does not show IOC limits)
    'tau': 1.5,
    'far_tol': 3.0,  # outside a shock window, two far inserts must agree within this to restart the grid
    'taker_obs': 0,
}


PARAMS = FV_PARAMS | {
    # ---- calm regime
    'cap': 100,
    'theta': 0.30,
    'theta_sd': 0.0,
    'theta_keep': 0.10,
    'keep_depth': 1,
    'reach_off': 1.0,
    'take': 0.15,
    'take_sd': 2.0,
    'max_sd': 0.6,  # no new quotes while the posterior sd is above this
    'inv_skew': 0.0,
    'size_top': 1000,
    'min_post': 1,
    'fill_room': 0,  # mech: top up the target level when unused room >= this (0 = off)
    'pj_q': 0,  # mech: pre-join size one tick behind target (0 = off)
    'pj_reach': 1,
    'trade_start_us': 0,
    # ---- regimes. "burst": the filter's jump excitation exc > burst_exc (a jump in the last few s);
    # "stretch": at least stretch_n jumps in the trailing stretch_win s (a turbulent stretch between jumps).
    # Either one switches every key below from its calm value to the *_t value.
    'burst_exc': 0.05,
    'stretch_win': 60.0,
    'stretch_n': 0,  # 0 = stretch detection off
    'rv_hl': 10.0,  # s, EWMA half-life of the realised volatility of F (sampled every 100 ms)
    'rv_thr': 0.0,  # >0: turbulent while that volatility exceeds this (ticks / sqrt s); round-2 findings use 2
    'passive_t': 0,  # 0: no passive quotes in turbulence (resting ones are cancelled)
    'cap_t': 100,
    'theta_t': 1.8,
    'theta_keep_t': 1.6,
    'take_t': 0.15,
    'take_sd_t': 2.0,
    'reach_t': 2.5,  # volatile takers price at round(S +- 2.55): orders within this of F stay reachable
    # take_q > 0: take only where the take_q-quantile of S clears the price by take_qm (sells: the 1-take_q
    # quantile), instead of the mean rule (mean - take - take_sd * sd). Same keys in both regimes.
    'take_q': 0.0,
    'take_qm': 0.0,
    # ---- unwind: with |pos| > unwind_pos, IOC back toward unwind_to at prices up to unwind_u (+ unwind_sd * sd)
    # worse than F. Frees the position limit for the next mispricing (the cap is the binding resource).
    'unwind_u': -1.0,  # < 0 disables
    'unwind_sd': 0.0,
    'unwind_pos': 50,
    'unwind_to': 0,
    'unwind_turb': 0,  # 1: also unwind during bursts
    # ---- volatile regime of the generator: turbulent while the filter's P(volatile) > pv_thr (> 1 disables)
    'pv_thr': 0.5,
    # ---- scheduled shocks (round 2: 27 per 6000 s): a 3.0 s maker drought, then S jumps by N(0, 43).
    # Drought = fewer than sg_thr market inserts in the trailing sg_win s. From detection on: no passive quotes
    # and the drought maker law; between detection + shock_lead and + shock_lead + shock_len the filter expects
    # the jump (rho_shock, one far insert restarts the grid). The window closes once the jump is detected.
    'shock': 1,
    'sg_win': 0.5,
    'sg_thr': 50,
    'shock_lead': 2.0,
    'shock_len': 2.5,
    'shock_cool': 20.0,  # s after a detection before the next can start
    'shock_flat': 0,  # 1: on detection, IOC the position flat at the touch
}

BUY, SELL = Side.BUY, Side.SELL


class FV:
    """Joint posterior of (regime, S): S on the grid (c + k) * h, k = -K..K (c an integer), regime as P(volatile).

    The S posterior is shared by both regimes (a factorised approximation): each maker insert multiplies it by the
    regime mixture of the two likelihoods, and updates P(volatile) by the ratio of the two evidences. Volatile makers
    never quote within 0.70 of S (calm: 74 % do), so the regime shows up within a few inserts.
    Prices are integers and h = 1/n, so every likelihood is a slice of a table precomputed per (law, side, clamped).
    Taker impact moves the origin c by whole cells; the sub-cell rest `r` is carried in `mean` only (<= h/2).
    """

    def __init__(self, p: dict, s0: float) -> None:
        self.p = p
        h = self.h = p['h']
        self.inv_h = round(1 / h)
        K = self.K = p['K']
        n = 2 * K + 1
        self.k = (np.arange(n) - K) * h
        self.k2 = self.k * self.k
        self.c = round(s0 / h)
        self.r = 0.0
        self.w = np.full(n, 1.0 / n)
        self.t = None
        self.exc = 0.0
        self.mean = float(s0)
        self.sd = 1.0
        self.far = None  # (t, implied S, is_bid, price) of a pending far insert
        self.jumps = 0
        self.ll = 0.0
        self.n_ll = 0
        self.two = p['regimes'] == 2
        self.pv = 0.0  # P(volatile)
        self.drought = False  # set by the strategy: maker inserts follow the drought law
        self.hot = False  # set by the strategy: inside an expected-shock window
        J = self.J = 2 * K + 4 * self.inv_h
        x = (np.arange(2 * J + 1) - J) * h
        eta = p['eta']

        def tables(d0: float, lam: float) -> tuple:
            F = -np.expm1(-np.maximum(x - d0, 0.0) / lam)
            F1 = -np.expm1(-np.maximum(x - 1.0 - d0, 0.0) / lam)
            cl = (1 - eta) * F + eta / 30.0
            un = (1 - eta) * (F - F1) + eta / 30.0
            # [bid unclamped, bid clamped, ask unclamped, ask clamped]
            return un, cl, un[::-1].copy(), cl[::-1].copy()

        self.T_c = tables(p['d0'], p['lam'])
        self.T_v = tables(p['d0_v'], p['lam_v'])
        self.T_d = tables(p['d0_d'], p['lam_d'])
        self.inv2h2 = 1.0 / (2 * h * h)
        self.vj = 2 * p['bj_v'] ** 2 * p['lamj_v']  # variance rate of the volatile residual jumps
        self.kern: dict[int, np.ndarray] = {}

    def predict(self, t: float) -> None:
        if self.t is None:
            self.t = t
            return
        dt = t - self.t
        if dt <= 0:
            return
        self.t = t
        p = self.p
        w = self.w
        pv = self.pv
        if self.two:
            pv += (p['q_cv'] * (1 - pv) - p['q_vc'] * pv) * dt
            self.pv = pv = min(max(pv, 1e-4), 1 - 1e-4)
        a = (p['sigma'] ** 2 + pv * self.vj) * dt * self.inv2h2
        while a > 1e-4:
            b = a if a < 0.45 else 0.45
            w[1:-1] = (1 - 2 * b) * w[1:-1] + b * (w[:-2] + w[2:])
            a -= b
        if self.exc:
            self.exc *= 0.5 ** (dt / p['exc_hl'])
            if self.exc < 1e-4:
                self.exc = 0.0
        pi = 1.0 - math.exp(-((p['rho_shock'] if self.hot else p['rho0']) + self.exc) * dt)
        w *= 1.0 - pi
        w += pi / w.size

    def shift(self, d: float) -> None:
        r = self.r + d
        s = round(r * self.inv_h)
        if s:
            self.c += s
            r -= s * self.h
        self.r = r
        self.mean += d

    def _blur(self, var: float) -> None:
        """Convolve the S posterior with N(0, var) (the impact noise of one filled taker)."""
        key = round(var * 2000)
        kern = self.kern.get(key)
        if kern is None:
            s = math.sqrt(max(var, 1e-9)) * self.inv_h
            m = max(1, math.ceil(4 * s))
            x = np.arange(-m, m + 1)
            kern = np.exp(-0.5 * (x / s) ** 2) if s > 0.05 else (x == 0).astype(float)
            kern /= kern.sum()
            self.kern[key] = kern
        if kern.size > 1:
            self.w = np.convolve(self.w, kern, 'same')

    def _post(self, post: np.ndarray, z: float, count_ll: bool) -> None:
        if z <= 1e-300:
            post = np.full(post.size, 1.0 / post.size)
            z = 1e-300
        else:
            post *= 1.0 / z
        self.w = post
        if count_ll:
            self.ll += math.log(z)
            self.n_ll += 1
        m = float(post @ self.k)
        v = float(post @ self.k2) - m * m
        old, old_sd = self.mean, self.sd
        self.mean = self.c * self.h + self.r + m
        self.sd = math.sqrt(v) if v > 1e-8 else 1e-4
        if abs(m) > 3.0:
            self._recentre(round(m * self.inv_h))
        if abs(self.mean - old) > self.p['jump_det'] and old_sd < self.p['jump_sd']:
            self._jumped()

    def _jumped(self) -> None:
        self.exc = min(self.exc + self.p['kick'], self.p['exc_max'])
        self.jumps += 1

    def _recentre(self, s: int) -> None:
        w = self.w
        n = w.size
        if abs(s) >= n:
            w[:] = 1.0 / n
        elif s > 0:
            w[:-s] = w[s:].copy()
            w[-s:] = 0.0
        else:
            w[-s:] = w[:s].copy()
            w[:-s] = 0.0
        self.c += s
        z = w.sum()
        if z > 0:
            w /= z

    def _slice(self, T: tuple, is_bid: bool, price: int, clamped: bool):
        """Likelihood over the grid from one law's tables, or None if the insert is beyond them.
        bids: x_k = (c + k - P) h -> index c - P + k + J; asks: x_k = (P - c - k) h -> same index, reversed table."""
        K = self.K
        i = self.c - price * self.inv_h - K + self.J
        tab = T[(0 if is_bid else 2) + clamped]
        if i < 0 or i + 2 * K + 1 > tab.size:
            return None
        return tab[i:i + 2 * K + 1]

    def _observe(self, is_bid: bool, price: int, clamped: bool, count_ll: bool) -> None:
        w = self.w
        if self.drought or not self.two:
            L = self._slice(self.T_d if self.drought else self.T_c, is_bid, price, clamped)
            if L is None:
                return
            post = w * L
            self._post(post, float(post.sum()), count_ll)
            return
        Lc = self._slice(self.T_c, is_bid, price, clamped)
        Lv = self._slice(self.T_v, is_bid, price, clamped)
        if Lc is None:
            return
        pc_ = w * Lc
        pv_ = w * Lv
        zc = float(pc_.sum())
        zv = float(pv_.sum())
        pv = self.pv
        z = (1 - pv) * zc + pv * zv
        if z > 1e-300:
            self.pv = min(max(pv * zv / z, 1e-4), 1 - 1e-4)
        pc_ *= 1 - pv
        pv_ *= pv
        pc_ += pv_
        self._post(pc_, z, count_ll)

    def maker(self, t: float, is_bid: bool, price: int, opp: int | None, count_ll: bool = True) -> None:
        """A market limit insert at `price`; `opp` is the best market price on the other side before it."""
        self.predict(t)
        clamped = opp is not None and (price == opp - 1 if is_bid else price == opp + 1)
        if not clamped:
            imp = price + 0.55 if is_bid else price - 0.55
            if abs(imp - self.mean) > self.K * self.h - 2:
                f = self.far
                if self.hot or (f is not None and t - f[0] < 0.2 and abs(f[1] - imp) < self.p['far_tol']):
                    # restart the grid at the new level, then apply the far insert(s)
                    obs = [(is_bid, price)] if self.hot or f is None else [(f[2], f[3]), (is_bid, price)]
                    self.c = round((imp if len(obs) == 1 else (f[1] + imp) / 2) * self.inv_h)
                    self.r = 0.0
                    self.w = np.full(self.w.size, 1.0 / self.w.size)
                    self.far = None
                    self.sd = 1.0  # not a "confident" detection: count the jump here
                    self._jumped()
                    for b, pr in obs:
                        self._observe(b, pr, False, False)
                else:
                    self.far = (t, imp, is_bid, price)
                return
        self._observe(is_bid, price, clamped, count_ll)

    def taker(self, t: float, is_buy: bool, price, filled: int) -> None:
        self.predict(t)
        p = self.p
        if p['taker_obs'] and price is not None:
            S = self.c * self.h + self.r + self.k
            o = (price - S) if is_buy else (S - price)
            L = p['w_u'] * ((o >= 0.5) & (o < 1.5))
            L = L + (1 - p['w_u']) / p['tau'] * np.exp(-np.maximum(o - 0.5, 0.0) / p['tau']) * (o >= 0.5) + 0.01
            post = self.w * L
            self._post(post, float(post.sum()), False)
        if filled:
            pv = self.pv
            d = (1 - pv) * p['c'] * filled ** p['e'] + pv * p['c_v'] * filled ** p['e_v']
            self.shift(d if is_buy else -d)
            var = (1 - pv) * p['jv'] + pv * p['jv_v']
            if var > 0:
                self._blur(var)

    def quantile(self, q: float) -> float:
        """q-quantile of S (cell resolution)."""
        cw = np.cumsum(self.w)
        i = int(np.searchsorted(cw, q * cw[-1]))
        return self.c * self.h + self.r + self.k[min(i, self.k.size - 1)]


def _newer(a: str, b: str) -> bool:
    return len(a) > len(b) or (len(a) == len(b) and a > b)


class FVMM(BaseStrategy):
    def __init__(self) -> None:
        super().__init__()
        self.p = PARAMS
        self.f: FV | None = None
        self.pos = 0
        self.orders: dict[str, list] = {}  # id -> [side, price, remaining]
        self.cancelling: set[str] = set()
        self.max_id = ''
        self.mbb: int | None = None  # market-only touch before the current event
        self.mba: int | None = None
        self.nb = self.na = 0
        self.now_t = 0.0
        self.last_new_t = 0.0
        self.jump_t: deque = deque()
        self.n_jumps = 0
        self.turb = False
        self.ins_t: deque = deque()  # times of the market inserts seen in the trailing sg_win
        self.sg_t = None  # drought detection time of the current shock
        self.sg_jumps = 0
        self.sg_next = 0.0
        self.t0 = None
        self.n_shocks = 0
        self.flat_req = False
        self.rv = 0.0  # EWMA of squared 100 ms changes of F, scaled to ticks^2 / s
        self.rv_F = None
        self.rv_next = 0.0

    # ---------------------------------------------------------------- private feed
    def on_private_order_update(self, event: PrivateOrderEvent) -> None:
        o = self.orders.get(event.order_id)
        if o is None:
            return
        st = event.status
        if st is OrderStatus.PARTIALLY_FILLED or st is OrderStatus.FILLED:
            self.pos += event.filled_quantity if o[0] is BUY else -event.filled_quantity
            o[2] = event.remaining_quantity
        if st is OrderStatus.FILLED or st is OrderStatus.CANCELED or st is OrderStatus.REJECTED:
            self.orders.pop(event.order_id, None)
            self.cancelling.discard(event.order_id)

    # ---------------------------------------------------------------- public feed
    def _scan(self, views) -> tuple[int | None, object]:
        """Market-only touch of one side and the newest market order in it (None if none is new).

        Market ids are equal-length digit strings (7 digits on every tape so far), so plain string order is
        id order; user ids ('u...') compare greater and are dropped by the owner check. If the id length ever
        grows, on_order_book_change notices that no insert was seen for a second and re-syncs max_id.
        """
        top = None
        for v in views:
            if v.owner == 'market':
                top = v.price
                break
        new = None
        mx = self.max_id
        for v in views:
            if v.order_id > mx and v.owner == 'market':
                mx = v.order_id
                new = v
        if new is not None:
            self.max_id = mx
            self.last_new_t = self.now_t
        return top, new

    def _resync(self, views) -> None:
        best = self.max_id
        for v in views:
            if v.owner == 'market' and _newer(v.order_id, best):
                best = v.order_id
        self.max_id = best

    def on_order_book_change(self, snapshot: OrderBookSnapshot, trades: list[Trade]) -> None:
        p = self.p
        bids, asks = snapshot.bids, snapshot.asks
        t = snapshot.timestamp_us * 1e-6
        self.now_t = t
        if self.t0 is None:
            self.t0 = t
        f = self.f
        if f is None:
            if not bids or not asks:
                return
            f = self.f = FV(p, (bids[0].price + asks[0].price) / 2)
        q = 0
        sgn = 0
        tid = None
        for tr in trades:
            if tr.taker_owner == 'market':
                q += tr.quantity
                sgn = 1 if tr.book_side == 'ask' else -1
                tid = tr.taker_order_id
        # new market inserts: only scan a side whose view list grew or whose top moved
        nb, na = len(bids), len(asks)
        scan_b = nb > self.nb or (nb and bids[0].price != self.mbb)
        scan_a = na > self.na or (na and asks[0].price != self.mba)
        self.nb, self.na = nb, na
        pbb, pba = self.mbb, self.mba
        newb = newa = None
        if scan_b:
            self.mbb, newb = self._scan(bids)
        if scan_a:
            self.mba, newa = self._scan(asks)
        if t - self.last_new_t > 1.0:  # ~180 inserts / s: a silent second means max_id is out of sync
            self._resync(bids)
            self._resync(asks)
            self.last_new_t = t
        if q:
            if self._newer_id(tid) and len(tid) == len(self.max_id):
                self.max_id = tid
            rested = (newb is not None and newb.order_id == tid) or (newa is not None and newa.order_id == tid)
            if not rested:  # a crossing limit order (only possible against us) is no IOC: no impact
                f.taker(t, sgn > 0, None, q)
        elif newb is not None:
            f.maker(t, True, newb.price, pba)
            self.ins_t.append(t)
        elif newa is not None:
            f.maker(t, False, newa.price, pbb)
            self.ins_t.append(t)
        if not bids or not asks or snapshot.timestamp_us < p['trade_start_us']:
            return
        self._quote(f, bids[0].price, asks[0].price, bids, asks)

    def _newer_id(self, oid: str | None) -> bool:
        return oid is not None and _newer(oid, self.max_id)

    # ---------------------------------------------------------------- trading
    def _send(self, side: Side, price: int, qty: int, kind: OrderType) -> None:
        oid = self.send_order(OrderRequest(side=side, price=price, quantity=qty, order_type=kind))
        self.orders[oid] = [side, price, qty]
        if kind is OrderType.IOC:
            self.cancelling.add(oid)  # counts against the room until it is done, never "kept"

    def _shock(self, f: FV) -> bool:
        """Drought / expected-shock state machine; True from drought detection until the shock window closes."""
        p = self.p
        t = self.now_t
        it = self.ins_t
        while it and it[0] < t - p['sg_win']:
            it.popleft()
        if self.sg_t is None:
            if t - self.t0 > p['sg_win'] and len(it) < p['sg_thr'] and t > self.sg_next:
                self.sg_t = t
                self.sg_jumps = f.jumps
                self.sg_next = t + p['shock_cool']
                f.drought = True
                self.n_shocks += 1
                if p['shock_flat'] and self.pos:
                    self.flat_req = True
            else:
                return False
        rel = t - self.sg_t
        jumped = f.jumps != self.sg_jumps and rel > p['shock_lead'] - 0.5
        if jumped or rel > p['shock_lead'] + p['shock_len']:
            self.sg_t = None
            f.drought = f.hot = False
            return False
        f.hot = rel >= p['shock_lead']
        return True

    def _turbulent(self, f: FV) -> bool:
        p = self.p
        if f.exc > p['burst_exc'] or f.pv > p['pv_thr']:
            return True
        if p['rv_thr']:
            t = self.now_t
            if t >= self.rv_next:
                if self.rv_F is not None:
                    d = f.mean - self.rv_F
                    self.rv += (1 - 0.5 ** ((t - self.rv_next + 0.1) / p['rv_hl'])) * (d * d * 10 - self.rv)
                self.rv_F = f.mean
                self.rv_next = t + 0.1
            if self.rv > p['rv_thr'] ** 2:
                return True
        if p['stretch_n']:
            jt = self.jump_t
            if f.jumps != self.n_jumps:
                self.n_jumps = f.jumps
                jt.append(self.now_t)
            while jt and jt[0] < self.now_t - p['stretch_win']:
                jt.popleft()
            return len(jt) >= p['stretch_n']
        return False

    def _quote(self, f: FV, bb: int, ba: int, bids, asks) -> None:
        p = self.p
        F, sd = f.mean, f.sd
        turb = self._turbulent(f)
        if p['shock'] and self._shock(f):
            turb = True
        self.turb = turb
        if turb:
            cap, theta, keep, take, take_sd = p['cap_t'], p['theta_t'], p['theta_keep_t'], p['take_t'], p['take_sd_t']
        else:
            cap, theta, keep, take, take_sd = p['cap'], p['theta'], p['theta_keep'], p['take'], p['take_sd']
        pos = self.pos
        rest_b = rest_a = 0
        for o in self.orders.values():
            if o[0] is BUY:
                rest_b += o[2]
            else:
                rest_a += o[2]
        # ---- take stale quotes
        if self.flat_req:
            self.flat_req = False
            if pos > 0 and pos - rest_a > 0:
                self._send(SELL, max(1, bb - 3), pos - rest_a, OrderType.IOC)
                rest_a += pos - rest_a
            elif pos < 0 and -pos - rest_b > 0:
                self._send(BUY, ba + 3, -pos - rest_b, OrderType.IOC)
                rest_b += -pos - rest_b
        tq = p['take_q']
        if tq:
            # cheap prefilter on the mean, then the quantile only for a candidate
            lo_b = self.f.quantile(tq) - p['take_qm'] if take and ba < F else -1e18
            hi_a = self.f.quantile(1 - tq) + p['take_qm'] if take and bb > F else 1e18
        else:
            tk = take + take_sd * sd
            lo_b, hi_a = F - tk, F + tk
        if take and ba <= lo_b:
            lim = math.floor(lo_b)
            q = 0
            for o in asks:
                if o.price > lim:
                    break
                if o.owner == 'market':
                    q += o.quantity
            q = min(q, (p['xcap'] if p['xcap'] and ba <= F - p['xedge'] else cap) - pos - rest_b)
            if q > 0:
                self._send(BUY, lim, q, OrderType.IOC)
                rest_b += q
        if take and bb >= hi_a:
            lim = math.ceil(hi_a)
            q = 0
            for o in bids:
                if o.price < lim:
                    break
                if o.owner == 'market':
                    q += o.quantity
            q = min(q, (p['xcap'] if p['xcap'] and bb >= F + p['xedge'] else cap) + pos - rest_a)
            if q > 0:
                self._send(SELL, lim, q, OrderType.IOC)
                rest_a += q
        # ---- unwind toward flat
        if p['unwind_u'] >= 0 and (not turb or p['unwind_turb']):
            u = p['unwind_u'] + p['unwind_sd'] * sd
            if pos - rest_a > p['unwind_pos'] and bb >= F - u:
                lim = max(bb - 5, math.ceil(F - u))
                q = 0
                for o in bids:
                    if o.price < lim:
                        break
                    if o.owner == 'market':
                        q += o.quantity
                q = min(q, pos - rest_a - p['unwind_to'])
                if q > 0:
                    self._send(SELL, lim, q, OrderType.IOC)
                    rest_a += q
            elif -pos - rest_b > p['unwind_pos'] and ba <= F + u:
                lim = min(ba + 5, math.floor(F + u))
                q = 0
                for o in asks:
                    if o.price > lim:
                        break
                    if o.owner == 'market':
                        q += o.quantity
                q = min(q, -pos - rest_b - p['unwind_to'])
                if q > 0:
                    self._send(BUY, lim, q, OrderType.IOC)
                    rest_b += q
        # ---- passive quotes
        wide = p['theta_sd'] * sd
        sk = p['inv_skew'] * pos / p['cap']
        th_b, th_a = theta + wide + sk, theta + wide - sk
        kp_b, kp_a = keep + wide + sk, keep + wide - sk
        passive = not turb or p['passive_t']
        tb = min(math.floor(F - th_b), ba - 1)
        ta = max(math.ceil(F + th_a), bb + 1)
        kd = p['keep_depth']
        ro = p['reach_t'] if turb else p['reach_off']
        rb = round(F - ro)  # deepest bid the next sell taker reaches
        ra = round(F + ro)
        have_b = have_a = False
        pjr = p['pj_reach'] if p['pj_q'] and not turb else 0
        for oid, o in self.orders.items():
            if oid in self.cancelling:
                continue
            px = o[1]
            if not passive:
                pass
            elif o[0] is BUY:
                if F - px >= kp_b and tb - kd <= px < ba and px >= rb - pjr:
                    have_b = have_b or px == tb
                    continue
            else:
                if px - F >= kp_a and ta + kd >= px > bb and px <= ra + pjr:
                    have_a = have_a or px == ta
                    continue
            self.cancelling.add(oid)
            self.cancel_order(oid)
        if sd > p['max_sd'] or not passive:
            return
        pj = p['pj_q'] if not turb else 0
        if pj:
            # pre-join: also rest pj_q one tick behind the target, to own queue seniority there when F moves
            hb = any(o[0] is BUY and o[1] == tb - 1 for oid, o in self.orders.items() if oid not in self.cancelling)
            ha = any(o[0] is SELL and o[1] == ta + 1 for oid, o in self.orders.items() if oid not in self.cancelling)
            if not hb and tb - 1 >= rb - p['pj_reach']:
                q = min(cap - pos - rest_b, pj)
                if q >= p['min_post']:
                    self._send(BUY, tb - 1, q, OrderType.LIMIT)
                    rest_b += q
            if not ha and ta + 1 <= ra + p['pj_reach']:
                q = min(cap + pos - rest_a, pj)
                if q >= p['min_post']:
                    self._send(SELL, ta + 1, q, OrderType.LIMIT)
                    rest_a += q
        if not have_b or (p['fill_room'] and cap - pos - rest_b >= p['fill_room']):
            q = min(cap - pos - rest_b, p['size_top'])
            if q >= p['min_post']:
                self._send(BUY, tb, q, OrderType.LIMIT)
        if not have_a or (p['fill_room'] and cap + pos - rest_a >= p['fill_room']):
            q = min(cap + pos - rest_a, p['size_top'])
            if q >= p['min_post']:
                self._send(SELL, ta, q, OrderType.LIMIT)


# ======================================================================================================
# fvs_mm: fv_mm (snapshot) + a time-locked shock sniper (strategies/peter/round2_findings.md, "shock sniper").
#
# The shock jump lands 3.00 s after the drought starts, while the stale side of the thin drought book is still
# resting (the first market taker in the jump direction comes a median ~50 ms later). For big jumps the new
# level itself is outside the 20-tick public depth, but the makers' clipping gives it away: after an up-jump
# every new bid is clipped to (stale ask - 1), several ticks above the old best bid of the wide drought spread.
# So inside [sg_t + ss_t0, sg_t + ss_t1] the first new market insert that is
#     a bid at (ask before - 1) with spread before >= ss_spread, or an ask >= ask before + ss_far   -> UP
#     an ask at (bid before + 1) with spread before >= ss_spread, or a bid <= bid before - ss_far  -> DOWN
# triggers IOCs through the stale side up to (stale touch +- ss_lim), ss_shots times ss_gap_ms apart, for the
# whole room. fv_mm's own quoting is suspended for ss_hold s; then it manages the position as usual
# (ss_unwind > 0: IOC the position back toward 0 at prices up to ss_unwind ticks worse than the touch after the
# hold, at most every 50 ms).
# ======================================================================================================
PARAMS.update({
    'ss_on': 1,
    'ss_t0': 0.8,  # s after drought detection (detection lags the onset by ~0.2-0.5 s; jump at onset + 3.0)
    'ss_t1': 4.0,
    'ss_spread': 3,
    'ss_far': 3,
    'ss_lim': 8,  # IOC limit: stale touch +- this
    'ss_shots': 3,
    'ss_gap_ms': 15,
    'ss_hold': 0.3,
    'ss_unwind': 0,
    'ss_trade': 0,  # 1: also trigger on a market taker that sweeps >= ss_trade_lv levels in the window
    'ss_trade_lv': 3,
    'ss_onset': 1,  # 1: window relative to the change-point estimate of the drought onset (MLE 180/s -> 54/s)
    'ss_w0': 2.99,  # window [onset + ss_w0, onset + ss_w1] (the jump lands at onset + 3.00)
    'ss_w1': 3.10,
    'ss_r1': 180.0,
    'ss_r2': 54.0,
    # ---- mech: deliberate short breaches. The penalty is 10k per second over |100| regardless of size, so
    # snipes may take up to xcap (IOC takes) / xs_cap (shock sniper) and the excess over xu_tgt is IOC'd back
    # at prices down to F -+ xu_lim at once. 0 = off (baseline).
    'xcap': 0,
    'xs_cap': 0,
    'xedge': 3.0,  # xcap applies only when the touch is at least this far through F
    'xu_tgt': 100,
    'xu_lim': 3.0,
    'xu_gap_ms': 6,
    'xu_on': 0,
    # ---- snipe.py (0/None = snipe800 behaviour)
    'ss_sticky': 0,  # keep the sniper armed ss_sticky_s after fv_mm closes its shock window (fast.py)
    'ss_sticky_s': 4.5,
    'xu_me': None,  # exit floor: excess is IOC'd back no worse than snipe avg entry +- xu_me (None = off)
    'xu_ts': 0.3,  # time stop (s after trigger): after it the floor is dropped and the excess goes at F -+ xu_lim2
    'xu_lim2': 6.0,
    'xu_block': 1,  # with xu_me: suspend fv_mm quoting while the excess is held under the floor
    'xu_pass': 0,  # with xu_me: also rest the excess as a limit at the floor (passive exit into the taker burst)
    'xs_conf': 0,
    'xs_edge': None,  # excess over cap only if the filter mean at trigger is >= this many ticks through the stale touch
    'xu_ts2': 0,  # >0: time stop (s after trigger) without floor: after it the exit IOCs go at F -+ xu_lim2
    'xs_k': 0,  # >0: excess size capped at xs_k * (edge - xs_edge) lots  # >0: shots after the first may exceed cap only if F moved >= xs_conf ticks past ss_ref
})


class FVAShock(FVMM):
    def __init__(self) -> None:
        super().__init__()
        self.ss_dir = 0
        self.ss_ref = 0
        self.ss_left = 0
        self.ss_next = 0.0
        self.ss_hold_until = -1.0
        self.ss_sg = None  # sg_t of the shock already fired on
        self.n_snipes = 0
        self.my_ins: deque = deque()
        self.ss_onset = None
        self.ss_onset_sg = None
        self.xu_ids: set = set()
        self.xu_next = 0.0
        self.ss_ids: set = set()
        self.ss_q = 0
        self.ss_c = 0.0
        self.ss_t = -1e9
        self.ss_last = None
        self.xp_id = None
        self.ss_first = True
        self.ss_edge = 0.0

    def on_private_order_update(self, event) -> None:
        if event.order_id in self.ss_ids and event.filled_quantity and event.fill_price is not None:
            self.ss_q += event.filled_quantity
            self.ss_c += event.filled_quantity * event.fill_price
        if event.order_id == self.xp_id and event.status in (OrderStatus.FILLED, OrderStatus.CANCELED,
                                                              OrderStatus.REJECTED):
            self.xp_id = None
        super().on_private_order_update(event)

    def _xfloor(self, long: bool):
        """Exit floor for the excess (None = no floor)."""
        p = self.p
        if p['xu_me'] is None or not self.ss_q or self.now_t - self.ss_t > p['xu_ts']:
            return None
        e = self.ss_c / self.ss_q
        return math.ceil(e + p['xu_me']) if long else math.floor(e - p['xu_me'])

    def _room_side(self, side: Side) -> int:
        rest = sum(o[2] for o in self.orders.values() if o[0] is side)
        cap = self.p['xs_cap'] or self.p['cap']
        return cap - (self.pos + rest) if side is BUY else cap + self.pos - rest

    def _onset(self, sg: float):
        """MLE change point of the insert rate (ss_r1 -> ss_r2) over the inserts seen in [sg - 1.5, sg]."""
        p = self.p
        ts = [x for x in self.my_ins if sg - 1.5 <= x <= sg]
        if len(ts) < 20:
            return None
        import math as _m
        r1, r2 = p['ss_r1'], p['ss_r2']
        a, n = ts[0], len(ts)
        best, arg = -1e18, None
        lr1, lr2 = _m.log(r1), _m.log(r2)
        for k in range(n):
            c = ts[k]
            ll = k * lr1 - r1 * (c - a) + (n - k) * lr2 - r2 * (sg - c)
            if ll > best:
                best, arg = ll, c
        return arg

    def _xunwind(self, snapshot) -> None:
        p = self.p
        t = self.now_t
        if t < self.xu_next:
            return
        pos = self.pos
        tgt = p['xu_tgt']
        F = self.f.mean
        if pos > tgt:
            fl = sum(o[2] for oid, o in self.orders.items() if o[0] is SELL and oid in self.cancelling
                     and oid in self.xu_ids)
            q = pos - tgt - fl
            if q <= 0:
                return
            fl0 = self._xfloor(True)
            lim = math.ceil(F - self._xlim(fl0))
            if fl0 is not None:
                lim = max(lim, fl0)
                self._xpass(SELL, lim, q)
            avail = 0
            for o in snapshot.bids:
                if o.price < lim:
                    break
                if o.owner == 'market':
                    avail += o.quantity
            if avail <= 0:
                return
            oid = self.send_order(OrderRequest(side=SELL, price=lim, quantity=q, order_type=OrderType.IOC))
            self.orders[oid] = [SELL, lim, q]
            self.cancelling.add(oid)
            self.xu_ids.add(oid)
            self.xu_next = t + p['xu_gap_ms'] * 1e-3
        elif pos < -tgt:
            fl = sum(o[2] for oid, o in self.orders.items() if o[0] is BUY and oid in self.cancelling
                     and oid in self.xu_ids)
            q = -pos - tgt - fl
            if q <= 0:
                return
            fl0 = self._xfloor(False)
            lim = math.floor(F + self._xlim(fl0))
            if fl0 is not None:
                lim = min(lim, fl0)
                self._xpass(BUY, lim, q)
            avail = 0
            for o in snapshot.asks:
                if o.price > lim:
                    break
                if o.owner == 'market':
                    avail += o.quantity
            if avail <= 0:
                return
            oid = self.send_order(OrderRequest(side=BUY, price=lim, quantity=q, order_type=OrderType.IOC))
            self.orders[oid] = [BUY, lim, q]
            self.cancelling.add(oid)
            self.xu_ids.add(oid)
            self.xu_next = t + p['xu_gap_ms'] * 1e-3

    def _xlim(self, fl0) -> float:
        p = self.p
        if p['xu_me'] is not None and fl0 is None:
            return p['xu_lim2']
        if p['xu_ts2'] and self.now_t - self.ss_t > p['xu_ts2']:
            return p['xu_lim2']
        return p['xu_lim']

    def _xpass(self, side, px, q) -> None:
        if not self.p['xu_pass']:
            return
        o = self.orders.get(self.xp_id) if self.xp_id else None
        if o is not None and self.xp_id not in self.cancelling and o[1] == px and o[0] is side:
            return
        if o is not None and self.xp_id not in self.cancelling:
            self.cancelling.add(self.xp_id)
            self.cancel_order(self.xp_id)
        q = min(q, 1000 - 1)
        if q <= 0:
            return
        oid = self.send_order(OrderRequest(side=side, price=px, quantity=q, order_type=OrderType.LIMIT))
        self.orders[oid] = [side, px, q]
        self.xp_id = oid

    def _find(self, views, oid):
        for v in views:
            if v.order_id == oid:
                return v
        return None

    def on_order_book_change(self, snapshot: OrderBookSnapshot, trades: list[Trade]) -> None:
        p = self.p
        pbb, pba, pmax = self.mbb, self.mba, self.max_id
        if self.sg_t is not None:
            self.ss_last = self.sg_t
        super().on_order_book_change(snapshot, trades)
        if p['xu_on'] and self.f is not None:
            self._xunwind(snapshot)
            if self.xp_id is not None and abs(self.pos) <= p['xu_tgt'] and self.xp_id not in self.cancelling:
                self.cancelling.add(self.xp_id)
                self.cancel_order(self.xp_id)
        if not p['ss_on'] or snapshot.timestamp_us < p['trade_start_us']:
            return
        t = self.now_t
        sg = self.sg_t
        if sg is not None:
            self.ss_last = sg
        elif p['ss_sticky'] and self.ss_last is not None and t - self.ss_last < p['ss_sticky_s']:
            sg = self.ss_last
        mi = self.my_ins
        if self.max_id != pmax:
            mi.append(t)
        while mi and mi[0] < t - 2.0:
            mi.popleft()
        if sg is not None and sg != self.ss_onset_sg:
            self.ss_onset_sg = sg
            self.ss_onset = self._onset(sg) if p['ss_onset'] else None
        if sg is None or self.ss_onset is None:
            in_win = sg is not None and p['ss_t0'] <= t - sg <= p['ss_t1']
        else:
            in_win = p['ss_w0'] <= t - self.ss_onset <= p['ss_w1']
        if self.ss_dir == 0 and sg is not None and sg != self.ss_sg and in_win \
                and pbb is not None and pba is not None:
            d = 0
            if self.max_id != pmax:
                v = self._find(snapshot.bids, self.max_id)
                if v is not None:
                    if v.price == pba - 1 and pba - pbb >= p['ss_spread']:
                        d = 1
                    elif v.price <= pbb - p['ss_far']:
                        d = -1
                else:
                    v = self._find(snapshot.asks, self.max_id)
                    if v is not None:
                        if v.price == pbb + 1 and pba - pbb >= p['ss_spread']:
                            d = -1
                        elif v.price >= pba + p['ss_far']:
                            d = 1
            if not d and p['ss_trade'] and trades:
                lv = {tr.price for tr in trades if tr.taker_owner == 'market'}
                if len(lv) >= p['ss_trade_lv']:
                    d = 1 if trades[-1].book_side == 'ask' else -1
            if d:
                self.ss_dir, self.ss_sg = d, sg
                self.ss_ref = pba if d > 0 else pbb
                self.ss_left = p['ss_shots']
                self.ss_next = t
                self.ss_hold_until = t + p['ss_hold']
                self.n_snipes += 1
                self.ss_q, self.ss_c, self.ss_t, self.ss_first = 0, 0.0, t, True
                self.ss_edge = (self.f.mean - self.ss_ref) * d
                for oid, o in list(self.orders.items()):  # pull our own stale quotes on the losing side
                    if oid not in self.cancelling and o[0] is (SELL if d > 0 else BUY):
                        self.cancelling.add(oid)
                        self.cancel_order(oid)
        if self.ss_dir and self.ss_left > 0 and t >= self.ss_next:
            side = BUY if self.ss_dir > 0 else SELL
            q = self._room_side(side)
            if p['xs_conf'] and p['xs_cap'] and not self.ss_first:
                mv = (self.f.mean - self.ss_ref) * self.ss_dir
                if mv < p['xs_conf']:
                    q -= p['xs_cap'] - p['cap']
            px = self.ss_ref + p['ss_lim'] if side is BUY else self.ss_ref - p['ss_lim']
            if p['xs_edge'] is not None and p['xs_cap']:
                ex = p['xs_cap'] - p['cap']
                if self.ss_edge < p['xs_edge']:
                    q -= ex
                elif p['xs_k']:
                    q -= max(0, ex - int(p['xs_k'] * (self.ss_edge - p['xs_edge'])))
            if p['xs_conf'] and p['xs_cap'] and self.ss_first:
                q -= p['xs_cap'] - p['cap']
            self.ss_first = False
            if q > 0 and px > 0:
                self._send(side, px, q, OrderType.IOC)
                self.ss_ids.add(next(reversed(self.orders)))
            self.ss_left -= 1
            self.ss_next = t + p['ss_gap_ms'] * 1e-3
        if self.ss_dir and t >= self.ss_hold_until and self.ss_left <= 0:
            if p['ss_unwind'] and self.pos and t >= self.ss_next:
                bids, asks = snapshot.bids, snapshot.asks
                if self.pos > 0 and bids:
                    self._send(SELL, bids[0].price - p['ss_unwind'], min(self.pos, self._room_side(SELL)), OrderType.IOC)
                elif self.pos < 0 and asks:
                    self._send(BUY, asks[0].price + p['ss_unwind'], min(-self.pos, self._room_side(BUY)), OrderType.IOC)
                self.ss_next = t + 0.05
                if abs(self.pos) > 0:
                    return
            self.ss_dir = 0

    def _quote(self, f, bb, ba, bids, asks) -> None:
        if self.p['ss_on'] and (self.ss_dir and (self.now_t < self.ss_hold_until or self.ss_left > 0)):
            return
        if self.p['xu_block'] and abs(self.pos) > self.p['xu_tgt'] and self._xfloor(self.pos > 0) is not None:
            return
        super()._quote(f, bb, ba, bids, asks)


# snipe800: shock sniper takes up to 800 lots (breaching the +-100 limit briefly) and IOCs the excess back
# at prices down to F -+ 6 ticks. Net of the 10k/s breach penalty. Source: strategies/marek/round2/hyp/mech.py.
PARAMS.update({"xu_on": 1, "xs_cap": 800, "xu_lim": 6.0})

# rebased on strategies/marek/round2/fillsnipe.py (mech.py + fill_room=5, xs_cap=950, xu_lim=1.0)
PARAMS.update({"fill_room": 5, "xu_on": 1, "xs_cap": 950, "xu_lim": 1.0})
