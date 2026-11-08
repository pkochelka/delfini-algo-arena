"""r4_mom: round-4 candidate = Marek's r3_dev_best + round-4 selection (end of file) + momentum flags (off).

Round 4 = round 3 + momentum in the volatile regime (strategies/peter/round4_findings.md). Results (r4_eval.py):
full r4 tape 1438.3k vs r3_dev_best 1365.9k (+72.3k, 9/12 pieces), wf folds +64.8k (t 2.8), holdout +12.8k (3/4).
  kept:   dr_margin 0.9 (drought quoting stops 2.1 s after the onset estimate, 0.9 s before the jump), lamj_v 1.0 /
          bj_v 1.2 (volatile residual jumps, refit on r4 by session db)
  failed (flags kept, off): momentum-forecast shift of the take reference (mk, -33k..-400k on wf even with the
          limit capped at the touch, m_sl), of the passive quote centre (mq), forecast veto of turbulent takes
          (m_veto, ~0). The forecast is calibrated, but dev_best's stale-quote takes already follow it and the
          +-100 cap binds ~90 % of turbulent time, so a shift can only chase or pay to flip.
  flat (|d| < 7k): theta, theta_keep, take, take_t, take_sd_t, pv_thr, burst_exc, vf_*, cu_* unload, dr_theta/keep.

--- r3_dev_best: round-3 strategy of the develop angle (marek/research/round3/ideas/develop.md).

r3_best.py (fv_mm S filter + stale-quote taker + shock sniper, round-3 ladder filter) plus, all behind PARAMS flags
(off = r3_best), the settings at the end of this file:
  - bigmoves C2: early shock trigger on a market sweep with edge credit (bm_sw*), clip-trigger edge credit (ss_clip_e 2)
  - calm c2: calm IOC take 0.10, reduce-only calm IOC at edge >= 0 + 1 sd (tr_take), theta_keep 0
  - volatile vfm20: momentum-gated flatten-only passive quote in turbulence (vf_*)
  - filter constants of the final s_dynamics model (calm c, e, xi, transient p / T; regime switch rates; sigma 0.15)
Round 3 853.7k (r3_best 795.0k), round 2 778.4k (761.0k), net of the breach penalty.

fv_mm: market maker + stale-quote taker around a Bayesian estimate of the generator's hidden price S.

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
    # flow: quote passively into the post-shock taker burst (ps_win s after the shock jump is detected)
    'ps_on': 0,
    'ps_t0': 0.0,
    'ps_win': 3.0,
    'ps_theta': 1.0,
    'ps_keep': 0.8,
    'ps_maxsd': 1.0,
    'ps_size': 1000,
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
        self.Y = 0.0  # round 3: transient part of the summed taker impact (decays to 0 with time constant T)
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
        if p['tr_on'] and self.Y:
            T = (1 - pv) * p['tr_T'] + pv * p['tr_T_v']
            dY = self.Y * -math.expm1(-dt / T)
            self.Y -= dY
            self.shift(-dY)
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
            if p['tr_on'] and not (p['tr_sr'] and (self.drought or self.hot)):
                fr = (1 - pv) * (1 - p['tr_p']) + pv * (1 - p['tr_p_v'])
                self.Y += fr * d if is_buy else -fr * d
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
        self.lad_last = {True: None, False: None}  # side -> (t_us, price) of the last market insert
        self.lad_pend: list = []  # deferred deep inserts [t, t_us, is_bid, price, opp]
        self.lad_rung = False  # the current event's insert is a ladder rung (or a deferred deep insert)
        self.n_lad = 0
        self.vx_t = {BUY: -1e9, SELL: -1e9}  # develop vx_*: last turbulent take sent per side
        self.cu_t = -1e9  # develop cu_*: time / sign / size of the last calm IOC take
        self.cu_s = 0
        self.cu_q = 0
        self.fh: deque = deque(maxlen=128)  # round 4: F sampled every 100 ms (momentum forecast)
        self.fh_next = 0.0
        self.mom = 0.0  # last momentum forecast (ticks over ~m_h s)

    # ---------------------------------------------------------------- round-4 momentum forecast
    # Round 4 = round 3 + momentum (round4_findings.md): in the volatile regime S keeps moving the way it just
    # moved for ~5-10 s. The market touch lags S, so F - mid is the best single proxy of S's velocity; recent F
    # changes add to it. m = E[F(t + h) - F(t)] from an OLS fit on the public tape (r4_alpha.py, turbulent rows).
    def _mom(self, F: float) -> float:
        p = self.p
        h = self.fh
        n = len(h)
        if n < 6 or self.mbb is None or self.mba is None:
            return 0.0
        d = lambda k: F - h[-k if n >= k else 0]  # noqa: E731  (F change over ~k * 100 ms)
        m = (p['mg'] * (F - 0.5 * (self.mbb + self.mba)) + p['m05'] * d(5) + p['m1'] * d(10) + p['m2'] * d(20)
             + p['m5'] * d(50) + p['m10'] * d(100))
        mx = p['m_max']
        return mx if m > mx else (-mx if m < -mx else m)

    # ---------------------------------------------------------------- round-3 ladder maker filter
    # Round 3 adds a "ladder" maker: bursts of 2-6 inserts exactly 2000 us apart, same side, one tick further out
    # each, at 2..7 ticks behind the same-side touch, placed relative to the touch (not S). Under the S-maker law
    # they look like wide (volatile-law) quotes and drag both S and P(volatile). lad_on: drop rung 2+ of every
    # chain, and hold any insert >= lad_deep ticks behind the touch for lad_wait us; drop it if a rung follows.
    def _lad_flush(self, t_us: int) -> None:
        if not self.lad_pend:
            return
        keep = []
        for e in self.lad_pend:
            if t_us - e[1] > self.p['lad_wait']:
                self.f.maker(e[0], e[2], e[3], e[4])
                self.ins_t.append(e[0])
            else:
                keep.append(e)
        self.lad_pend = keep

    def _lad_maker(self, t: float, t_us: int, is_bid: bool, price: int, own: int | None, opp: int | None) -> None:
        p = self.p
        last = self.lad_last[is_bid]
        self.lad_last[is_bid] = (t_us, price)
        self.lad_rung = False
        if last is not None and abs(t_us - last[0] - 2000) <= p['lad_tol'] \
                and price == last[1] + (-1 if is_bid else 1):
            self.lad_rung = True
            self.n_lad += 1
            self.lad_pend = [e for e in self.lad_pend if not (e[2] == is_bid and e[3] == last[1])]
            return
        if own is not None and (own - price if is_bid else price - own) >= p['lad_deep'] \
                and not (p['lad_dr'] and self.sg_t is not None):
            self.lad_rung = True
            self.lad_pend.append([t, t_us, is_bid, price, opp])
            return
        self.f.maker(t, is_bid, price, opp)
        self.ins_t.append(t)

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
        lad = p['lad_on']
        if lad:
            self.lad_rung = False
            self._lad_flush(snapshot.timestamp_us)
        if q:
            if self._newer_id(tid) and len(tid) == len(self.max_id):
                self.max_id = tid
            rested = (newb is not None and newb.order_id == tid) or (newa is not None and newa.order_id == tid)
            if not rested:  # a crossing limit order (only possible against us) is no IOC: no impact
                f.taker(t, sgn > 0, None, q)
        elif newb is not None:
            if lad:
                self._lad_maker(t, snapshot.timestamp_us, True, newb.price, pbb, pba)
            else:
                f.maker(t, True, newb.price, pba)
                self.ins_t.append(t)
        elif newa is not None:
            if lad:
                self._lad_maker(t, snapshot.timestamp_us, False, newa.price, pba, pbb)
            else:
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
                if p['tr_sr']:
                    f.Y = 0.0
                self.n_shocks += 1
                if p['shock_flat'] and self.pos:
                    self.flat_req = True
            else:
                return False
        rel = t - self.sg_t
        jumped = f.jumps != self.sg_jumps and rel > p['shock_lead'] - 0.5
        if jumped or rel > p['shock_lead'] + p['shock_len']:
            if jumped:
                self.ps_t = t
            self.sg_t = None
            f.drought = f.hot = False
            return False
        f.hot = rel >= p['shock_lead']
        return True

    def _dr_ok(self) -> bool:
        return False

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
        F, sd = f.mean - p['y_k'] * f.Y, f.sd
        if self.now_t >= self.fh_next:
            self.fh.append(f.mean)
            self.fh_next = self.now_t + 0.1
        turb = self._turbulent(f)
        drq = False
        if p['shock'] and self._shock(f):
            drq = self._dr_ok()
            turb = not drq
        ps = False
        if p['ps_on'] and self.ps_t is not None:
            r = self.now_t - self.ps_t
            if p['ps_t0'] <= r < p['ps_win'] and self.sg_t is None:
                ps = True
                turb = False
        self.turb = turb
        if turb:
            cap, theta, keep, take, take_sd = p['cap_t'], p['theta_t'], p['theta_keep_t'], p['take_t'], p['take_sd_t']
        else:
            cap, theta, keep, take, take_sd = p['cap'], p['theta'], p['theta_keep'], p['take'], p['take_sd']
        if ps:
            theta, keep = p['ps_theta'], p['ps_keep']
        if drq:
            cap, theta, keep = p['dr_cap'], p['dr_theta'], p['dr_keep']
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
        if p['cgk'] and not turb and self.mbb is not None and self.mba is not None:
            # develop: calm take reference shifted by the book-lag momentum k * (F - mid)
            gs = p['cgk'] * max(-p['cm_max'], min(p['cm_max'], F - 0.5 * (self.mbb + self.mba)))
            lo_b += gs
            hi_a += gs
        if p['ctk_y'] and not turb:
            # develop: calm takes judged against F - ctk_y * Y (Y = transient impact still to decay)
            dy = p['ctk_y'] * f.Y
            if p['ctk_strict']:  # only ever make a take harder (fade the transient on the side it inflates)
                lo_b -= max(dy, 0.0)
                hi_a -= min(dy, 0.0)
            else:
                lo_b -= dy
                hi_a -= dy
        if p['tr_take'] is not None and not turb and take:
            # calm angle (ideas/calm_a.py): reduce-only IOC at a lower threshold (flattening frees room)
            tkr = p['tr_take'] + p['tr_take_sd'] * sd
            if pos > 0 and bb >= F + tkr:
                lim = math.ceil(F + tkr)
                q = sum(o.quantity for o in bids if o.price >= lim and o.owner == 'market')
                q = min(q, pos - p['tr_to'], cap + pos - rest_a)
                if q > 0:
                    self._send(SELL, lim, q, OrderType.IOC)
                    rest_a += q
            elif pos < 0 and ba <= F - tkr:
                lim = math.floor(F - tkr)
                q = sum(o.quantity for o in asks if o.price <= lim and o.owner == 'market')
                q = min(q, -pos - p['tr_to'], cap - pos - rest_b)
                if q > 0:
                    self._send(BUY, lim, q, OrderType.IOC)
                    rest_b += q
        # ---- round 4: momentum. In turbulence (calm: mk_calm) both take references move by mk * forecast, so a
        # momentum entry is not undone by the opposite take; m_sl caps how far past the touch a shifted IOC may go.
        m = 0.0
        if p['mk'] and (turb or p['mk_calm']):
            m = self._mom(f.mean) * (p['mk'] if turb else p['mk_calm'])
            lo_b += m
            hi_a += m
        self.mom = m
        mv = self._mom(f.mean) if p['m_veto'] and turb else 0.0  # veto: no take against a strong forecast
        if take and ba <= lo_b and not (mv and mv < -p['m_veto']):
            lim = math.floor(lo_b)
            if m > 0:
                lim = min(lim, max(math.floor(lo_b - m), ba + p['m_sl']))
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
                if turb:
                    self.vx_t[BUY] = self.now_t
                if not turb:
                    self.cu_q = q + (self.cu_q if self.cu_s > 0 and self.now_t - self.cu_t < p['cu_t1'] else 0)
                    self.cu_t, self.cu_s = self.now_t, 1
        if take and bb >= hi_a and not (mv and mv > p['m_veto']):
            lim = math.ceil(hi_a)
            if m < 0:
                lim = max(lim, min(math.ceil(hi_a - m), bb - p['m_sl']))
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
                if turb:
                    self.vx_t[SELL] = self.now_t
                if not turb:
                    self.cu_q = q + (self.cu_q if self.cu_s < 0 and self.now_t - self.cu_t < p['cu_t1'] else 0)
                    self.cu_t, self.cu_s = self.now_t, -1
        # ---- develop vx_*: volatile IOC exit once the position is vx_age s old (the snipe markout is flat after ~2 s)
        if p['vx_u'] >= 0 and turb and abs(pos) >= p['vx_pos'] and self.sg_t is None:
            lng = pos > 0
            age = self.now_t - self.vx_t[BUY if lng else SELL]
            mom_ok = p['vx_mom'] is None or self.mbb is None or self.mba is None \
                or (F - 0.5 * (self.mbb + self.mba)) * (1 if lng else -1) <= p['vx_mom']
            if age >= p['vx_age'] and mom_ok:
                fl = sum(o[2] for oid, o in self.orders.items() if oid in self.cancelling and o[0] is (SELL if lng else BUY))
                want = abs(pos) - p['vx_to'] - fl
                u = p['vx_u']
                if lng and want > 0 and bb >= F - u:
                    lim = math.ceil(F - u)
                    q = min(sum(o.quantity for o in bids if o.price >= lim and o.owner == 'market'), want)
                    if q > 0:
                        self._send(SELL, lim, q, OrderType.IOC)
                        rest_a += q
                elif not lng and want > 0 and ba <= F + u:
                    lim = math.floor(F + u)
                    q = min(sum(o.quantity for o in asks if o.price <= lim and o.owner == 'market'), want)
                    if q > 0:
                        self._send(BUY, lim, q, OrderType.IOC)
                        rest_b += q
        # ---- develop cu_*: unload calm take inventory cu_t0..cu_t1 s after the take (takes peak at 3-5 s, then revert)
        if p['cu_u'] >= 0 and self.cu_s and not turb:
            age = self.now_t - self.cu_t
            if age > p['cu_t1'] or pos * self.cu_s <= 0:
                self.cu_s = 0
            elif age >= p['cu_t0']:
                u = p['cu_u'] + p['cu_sd'] * sd
                want = min(self.cu_q, pos * self.cu_s - p['cu_to'])
                # only in-flight IOCs count: the resting passive quote on the reducing side holds the whole room
                fl = sum(o[2] for oid, o in self.orders.items() if oid in self.cancelling
                         and o[0] is (SELL if self.cu_s > 0 else BUY))
                if self.cu_s > 0 and want - fl > 0 and bb >= F - u:
                    lim = math.ceil(F - u)
                    q = min(sum(o.quantity for o in bids if o.price >= lim and o.owner == 'market'), want - fl)
                    if q > 0:
                        self._send(SELL, lim, q, OrderType.IOC)
                        rest_a += q
                        self.cu_q -= q
                elif self.cu_s < 0 and want - fl > 0 and ba <= F + u:
                    lim = math.floor(F + u)
                    q = min(sum(o.quantity for o in asks if o.price <= lim and o.owner == 'market'), want - fl)
                    if q > 0:
                        self._send(BUY, lim, q, OrderType.IOC)
                        rest_b += q
                        self.cu_q -= q
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
        if p['cq_y'] and not turb:
            F = F - p['cq_y'] * f.Y  # develop: calm passive quotes around F - cq_y * Y
        if p['mq'] and turb:
            F = F + p['mq'] * self.mom  # round 4: turbulent passive quotes around the momentum-shifted F
        if p['cm_k'] and not turb and self.mbb is not None and self.mba is not None:
            # develop: calm book-lag momentum (S drifts toward F - mid: beta .24 / .44 / .63 at 1 / 2 / 5 s)
            g = F - 0.5 * (self.mbb + self.mba)
            F = F + p['cm_k'] * max(-p['cm_max'], min(p['cm_max'], g))
        wide = p['theta_sd'] * sd
        sk = p['inv_skew'] * pos / p['cap']
        if p['th_in'] is not None and not turb and self.mbb is not None and self.mba is not None \
                and self.mba - self.mbb >= 2:
            theta = p['th_in']  # develop: market spread >= 2: a quote inside it is first in the queue
        th_b, th_a = theta + wide + sk, theta + wide - sk
        kp_b, kp_a = keep + wide + sk, keep + wide - sk
        if p['th_red'] is not None and not turb and abs(pos) >= p['th_red_pos']:
            # develop: pinned in calm: the reducing quote at theta th_red (so it sits at the touch, not behind it)
            if pos > 0:
                th_a = min(th_a, p['th_red'])
            else:
                th_b = min(th_b, p['th_red'])
        passive = not turb or p['passive_t']
        # volatile angle (ideas/volatile_tk.py, vf_*): flatten-only passive quote in turbulence, momentum-gated
        vf = turb and p['vf_on'] and not p['passive_t']
        pos_gate = False
        if vf and p['vf_mom'] is not None and self.mbb is not None and self.mba is not None and pos:
            pos_gate = (F - 0.5 * (self.mbb + self.mba)) * (1 if pos > 0 else -1) > p['vf_mom']
        vf_block = vf and (abs(pos) < p['vf_pos'] or pos_gate or self.sg_t is not None
                           or (p['vf_shk'] and self.now_t < self.sg_next))
        if vf:
            passive = True
            th_b = th_a = p['vf_th']
            kp_b = kp_a = p['vf_keep']
        tb = min(math.floor(F - th_b), ba - 1)
        ta = max(math.ceil(F + th_a), bb + 1)
        kd = p['keep_depth']
        ro = p['reach_t'] if (turb or ps) else p['reach_off']
        rb = round(F - ro)  # deepest bid the next sell taker reaches
        ra = round(F + ro)
        have_b = have_a = False
        pjr = p['pj_reach'] if p['pj_q'] and not turb else 0
        for oid, o in self.orders.items():
            if oid in self.cancelling:
                continue
            px = o[1]
            if not passive or (vf and (vf_block or not ((o[0] is BUY and pos < 0) or (o[0] is SELL and pos > 0)))):
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
        if sd > (p['dr_maxsd'] if drq else (p['ps_maxsd'] if ps else p['max_sd'])) or not passive:
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
        if vf:
            if vf_block:
                return
            if pos < 0 and not have_b:
                q = min(-pos - rest_b, p['vf_q'])
                if q >= p['min_post']:
                    self._send(BUY, tb, q, OrderType.LIMIT)
            elif pos > 0 and not have_a:
                q = min(pos - rest_a, p['vf_q'])
                if q >= p['min_post']:
                    self._send(SELL, ta, q, OrderType.LIMIT)
            return
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
    'xs_reval': 0,  # 1: re-check the edge at every shot (max so far) so later shots can carry the excess
    'xs_late': 0,  # >0: no excess when the trigger is later than onset estimate + xs_late s (stale side gone)
    'xs_nostk': 0,  # 1: no excess on triggers that only fired thanks to ss_sticky
    'xs_m': None,  # excess IOC priced no worse than F -+ xs_m (None = same price as the cap shot)
    'xu_ts2': 0,  # >0: time stop (s after trigger) without floor: after it the exit IOCs go at F -+ xu_lim2
    'dr_on': 0,  # drought agent: keep quoting passively during the maker drought until onset_est + 3.0 - dr_margin
    'dr_margin': 0.3,
    'dr_fb': 0.4,  # onset fallback: detection - dr_fb when the change point is unavailable
    'dr_cap': 100,
    'dr_theta': 0.3,
    'dr_keep': 0.1,
    'dr_maxsd': 0.6,
    'dr_flat': 0,  # 1: at the cut, IOC the position toward 0 at prices down to F -+ dr_fu
    'dr_fu': 1.0,
    'xs_stk_edge': 0,
    'ss_inv': 0,  # shockgen: 1: a market event that changes nothing visible inside the jump window = an insert beyond
                  # the 20-tick depth (big jump); the first visible stale-side cancel within ss_inv_w then triggers
    'ss_inv_w': 0.015,
    'ss_cw_x': 0,  # 1: such early clipped triggers count as in-window for the excess (st_win)
    'ss_clip_e': 0,  # 1: early clipped triggers (ss_cw0 path) get edge credit xs_edge so the FIRST shot carries
                     # the excess; 2: every clipped-insert trigger does
    'ss_reclip': 0,  # 1: non-clip triggers before onset_est + ss_w1 are cap-only and a later clip trigger may re-fire
    'ss_e2': 2.0,  # ss_clip_e=3: clip credit only with F edge >= this or a same-side stale cancel in st_cw
    'ss_cw0': 0.0,  # shockgen: >0: the clipped-insert pattern alone (bid at ask-1 / ask at bid+1) may trigger from
                    # onset_est + ss_cw0 (onset estimate is sometimes ~50 ms late); other trigger paths keep ss_w0
    'ss_inv_n': 0,  # >0: need this many visible cancels on one side (none on the other) after the invisible event
    'ss_inv_edge': 10.0,  # edge credited to such a trigger (|J| > ~15), so it may carry the excess  # shockgen: >0: sticky triggers may still carry the excess when the edge is >= this
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
        self.ps_t = None
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
        self.ss_stk = False
        self.ss_inw = False
        self.dr_sg = None
        self.dr_est = None
        self.dr_flat_done = False
        self.dr_flat_pend = False
        self.st_cx: deque = deque()
        self.st_lta = -1e9
        self.st_ltb = -1e9
        self.st_pb: dict = {}
        self.st_pa: dict = {}
        self.inv_sig = None
        self.ss_kind = None
        self.ss_capo = False
        self.inv_t = -1e9
        self.st_vb = False
        self.st_va = False

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

    def _lad_far(self, a: int, b: int) -> bool:
        """A far insert (b - a ticks behind the touch) that may be a ladder rung: rungs never sit > 7 behind."""
        return self.lad_rung and b - a <= self.p['lad_band']

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
        if p['st_cn']:
            # cancel wave: visible market orders near the touch that vanished with no trade on that side
            cur_b = {v.order_id: v.price for v in snapshot.bids[:p['st_lv']] if v.owner == 'market'}
            cur_a = {v.order_id: v.price for v in snapshot.asks[:p['st_lv']] if v.owner == 'market'}
            tb = any(tr.book_side == 'bid' for tr in trades)
            ta = any(tr.book_side == 'ask' for tr in trades)
            cx = self.st_cx
            for tr in trades:
                if tr.taker_owner == 'market':
                    if tr.book_side == 'ask':
                        self.st_lta = t
                    else:
                        self.st_ltb = t
            self.st_vb = (not tb) and any(oid not in cur_b for oid in self.st_pb)
            self.st_va = (not ta) and any(oid not in cur_a for oid in self.st_pa)
            if not tb:
                for oid in self.st_pb:
                    if oid not in cur_b:
                        cx.append((t, -1))
            if not ta:
                for oid in self.st_pa:
                    if oid not in cur_a:
                        cx.append((t, 1))
            while cx and cx[0][0] < t - p['st_cw']:
                cx.popleft()
            self.st_pb, self.st_pa = cur_b, cur_a
        inv_hit = False
        if p['ss_inv'] and sg is not None:
            sig = hash((tuple((v.order_id, v.quantity) for v in snapshot.bids),
                        tuple((v.order_id, v.quantity) for v in snapshot.asks)))
            if in_win and sig == self.inv_sig and not trades and self.ss_onset is not None \
                    and t - self.ss_onset >= 2.995:
                self.inv_t = t
            self.inv_sig = sig
        cw_hit = False
        cl_hit = False
        rc_ok = bool(p['ss_reclip']) and self.ss_kind == 'other' and sg is not None and sg == self.ss_sg \
            and self.ss_onset is not None and t - self.ss_onset <= p['ss_w1'] and t - self.ss_t < 0.2
        arm = (self.ss_dir == 0 and sg is not None and sg != self.ss_sg) or rc_ok
        in_c = bool(p['ss_cw0']) and sg is not None and self.ss_onset is not None \
            and p['ss_cw0'] <= t - self.ss_onset <= p['ss_w1']
        if arm and in_c and not in_win \
                and pbb is not None and pba is not None and self.max_id != pmax and pba - pbb >= p['ss_spread']:
            d = 0
            v = self._find(snapshot.bids, self.max_id)
            if v is not None:
                if v.price == pba - 1:
                    d = 1
            else:
                v = self._find(snapshot.asks, self.max_id)
                if v is not None and v.price == pbb + 1:
                    d = -1
            if d:
                in_win = True
                cw_hit = True
        # ---- bigmoves: early sweep trigger. A market taker that trades at >= bm_sw_lv levels on one side (and none on
        # the other) from onset_est + bm_sw_t0 on counts as an in-window trigger; with bm_sw_cr a big sweep (side emptied
        # or price span >= bm_sw_span ticks; bm_sw_big_only: only such sweeps trigger early) gets edge credit bm_sw_edge, so the first shot may carry the excess (oversize).
        sw_d, sw_big = 0, False
        if p['bm_sw'] and trades and sg is not None and self.ss_onset is not None \
                and p['bm_sw_t0'] <= t - self.ss_onset <= p['ss_w1']:
            la = {tr.price for tr in trades if tr.taker_owner == 'market' and tr.book_side == 'ask'}
            lb = {tr.price for tr in trades if tr.taker_owner == 'market' and tr.book_side == 'bid'}
            if len(la) >= p['bm_sw_lv'] and not lb:
                sw_d = 1
                sw_big = max(la) - min(la) >= p['bm_sw_span'] or (p['bm_sw_empty'] and not any(v.owner == 'market' for v in snapshot.asks))
            elif len(lb) >= p['bm_sw_lv'] and not la:
                sw_d = -1
                sw_big = max(lb) - min(lb) >= p['bm_sw_span'] or (p['bm_sw_empty'] and not any(v.owner == 'market' for v in snapshot.bids))
            if sw_d and p['bm_sw_big_only'] and not sw_big:
                sw_d = 0
            if sw_d and not in_win and arm:
                in_win = True
        if arm and in_win and pbb is not None and pba is not None:
            d = 0
            if self.max_id != pmax:
                v = self._find(snapshot.bids, self.max_id)
                if v is not None:
                    if v.price == pba - 1 and pba - pbb >= p['ss_spread']:
                        d = 1
                        cl_hit = True
                    elif v.price <= pbb - p['ss_far'] and not (p['lad_ss'] and self._lad_far(v.price, pbb)):
                        d = -1
                else:
                    v = self._find(snapshot.asks, self.max_id)
                    if v is not None:
                        if v.price == pbb + 1 and pba - pbb >= p['ss_spread']:
                            d = -1
                            cl_hit = True
                        elif v.price >= pba + p['ss_far'] and not (p['lad_ss'] and self._lad_far(pba, v.price)):
                            d = 1
            if not d and p['ss_inv'] and 0 < t - self.inv_t <= p['ss_inv_w']:
                if p['ss_inv_n']:
                    na = sum(1 for (tc, sd) in self.st_cx if sd > 0 and tc > self.inv_t)
                    nb = sum(1 for (tc, sd) in self.st_cx if sd < 0 and tc > self.inv_t)
                    if nb >= p['ss_inv_n'] and na == 0:
                        d, inv_hit = -1, True
                    elif na >= p['ss_inv_n'] and nb == 0:
                        d, inv_hit = 1, True
                elif self.st_vb != self.st_va:
                    d = -1 if self.st_vb else 1
                    inv_hit = True
            if not d and p['st_cn'] and len(self.st_cx) >= p['st_cn']:
                na = sum(1 for (tc, sd) in self.st_cx if sd > 0)
                nb = len(self.st_cx) - na
                okt = not p['st_cnt']
                if na >= p['st_cn'] and nb == 0 and (okt or t - self.st_lta <= p['st_cw']):
                    d = 1
                elif nb >= p['st_cn'] and na == 0 and (okt or t - self.st_ltb <= p['st_cw']):
                    d = -1
            if not d and p['st_tq'] and trades:
                qa = sum(tr.quantity for tr in trades if tr.taker_owner == 'market' and tr.book_side == 'ask')
                qb = sum(tr.quantity for tr in trades if tr.taker_owner == 'market' and tr.book_side == 'bid')
                if qa >= p['st_tq'] and not qb:
                    d = 1
                elif qb >= p['st_tq'] and not qa:
                    d = -1
            if not d and p['ss_trade'] and trades:
                lv = {tr.price for tr in trades if tr.taker_owner == 'market'}
                if len(lv) >= p['ss_trade_lv']:
                    d = 1 if trades[-1].book_side == 'ask' else -1
            if not d and sw_d:
                d = sw_d
            if rc_ok and not cl_hit:
                d = 0
            if d:
                self.ss_kind = 'clip' if cl_hit else 'other'
                self.ss_capo = bool(p['ss_reclip']) and not cl_hit and self.ss_onset is not None \
                    and t - self.ss_onset <= p['ss_w1']
                self.ss_dir, self.ss_sg = d, sg
                self.ss_ref = pba if d > 0 else pbb
                self.ss_left = p['ss_shots']
                self.ss_next = t
                self.ss_hold_until = t + p['ss_hold']
                self.n_snipes += 1
                self.ss_q, self.ss_c, self.ss_t, self.ss_first = 0, 0.0, t, True
                self.ss_edge = (self.f.mean - self.ss_ref) * d
                if p['bm_sw_cr'] and sw_d == d and sw_big:
                    self.ss_edge = max(self.ss_edge, p['bm_sw_edge'])
                if inv_hit:
                    self.ss_edge = max(self.ss_edge, p['ss_inv_edge'])
                self.ss_stk = self.sg_t is None  # fired only thanks to ss_sticky
                self.ss_inw = self.ss_onset is not None and p['ss_w0'] - p['st_w'] <= t - self.ss_onset <= p['ss_w1']
                if cw_hit and p['ss_cw_x']:
                    self.ss_inw = True
                if p['bm_sw_cr'] and sw_d == d and sw_big:
                    self.ss_inw = True
                if p['ss_clip_e'] and cl_hit and (cw_hit or p['ss_clip_e'] > 1):
                    two = True
                    if p['ss_clip_e'] >= 3:
                        nd = sum(1 for (tc, sd) in self.st_cx if sd == d)
                        no = sum(1 for (tc, sd) in self.st_cx if sd == -d)
                        two = self.ss_edge >= p['ss_e2'] or (nd >= 1 and no == 0)
                    if two:
                        self.ss_edge = max(self.ss_edge, p['xs_edge'])
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
            if p['st_pm'] is not None and (self.f.mean - self.ss_ref) * self.ss_dir >= p['st_pe']:
                # price through every stale level that is still >= st_pm ticks inside F
                px = max(px, math.floor(self.f.mean - p['st_pm'])) if side is BUY else min(px, math.ceil(self.f.mean + p['st_pm']))
            if p['xs_edge'] is not None and p['xs_cap']:
                if p['xs_reval']:
                    self.ss_edge = max(self.ss_edge, (self.f.mean - self.ss_ref) * self.ss_dir)
                ex = p['xs_cap'] - p['cap']
                late = p['xs_late'] and (self.ss_onset is None or self.ss_t - self.ss_onset > p['xs_late'])
                if p['xs_nostk'] and self.ss_stk and not (p['st_win'] and self.ss_inw) \
                        and not (p['xs_stk_edge'] and self.ss_edge >= p['xs_stk_edge']):
                    late = True
                if self.ss_edge < p['xs_edge'] or late or self.ss_capo:
                    q -= ex
                elif p['xs_k']:
                    q -= max(0, ex - int(p['xs_k'] * (self.ss_edge - p['xs_edge'])))
            if p['xs_conf'] and p['xs_cap'] and self.ss_first:
                q -= p['xs_cap'] - p['cap']
            self.ss_first = False
            if p['xs_m'] is not None and p['xs_cap']:
                # the part of the shot beyond the normal cap only at prices >= xs_m ticks through F
                qn = max(0, min(q, p['cap'] - (self.pos + sum(o[2] for o in self.orders.values() if o[0] is side))
                                if side is BUY else
                                p['cap'] + self.pos - sum(o[2] for o in self.orders.values() if o[0] is side)))
                qx = q - qn
                F = self.f.mean
                pxx = min(px, math.floor(F - p['xs_m'])) if side is BUY else max(px, math.ceil(F + p['xs_m']))
                if qx > 0 and pxx != px and pxx > 0:
                    self._send(side, pxx, qx, OrderType.IOC)
                    self.ss_ids.add(next(reversed(self.orders)))
                    q = qn
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

    def _dr_ok(self) -> bool:
        p = self.p
        if not p['dr_on'] or self.ss_dir:
            return False
        sg = self.sg_t
        if sg != self.dr_sg:
            self.dr_sg = sg
            self.dr_est = self._onset(sg)
            self.dr_flat_done = False
        t = self.now_t
        on = self.dr_est if self.dr_est is not None else sg - p['dr_fb']
        cut = on + 3.0 - p['dr_margin']
        if t < cut:
            return True
        if p['dr_flat'] and not self.dr_flat_done and self.f is not None:
            self.dr_flat_done = True
            self.dr_flat_pend = True
        return False

    def _quote(self, f, bb, ba, bids, asks) -> None:
        if self.p['ss_on'] and (self.ss_dir and (self.now_t < self.ss_hold_until or self.ss_left > 0)):
            return
        if self.p['xu_block'] and abs(self.pos) > self.p['xu_tgt'] and self._xfloor(self.pos > 0) is not None:
            return
        super()._quote(f, bb, ba, bids, asks)
        if self.dr_flat_pend:
            self.dr_flat_pend = False
            F, pos, u = f.mean, self.pos, self.p['dr_fu']
            if pos > 0:
                lim = math.ceil(F - u)
                q = sum(o.quantity for o in bids if o.price >= lim and o.owner == 'market')
                q = min(q, pos, self._room_side(SELL))
                if q > 0:
                    self._send(SELL, lim, q, OrderType.IOC)
            elif pos < 0:
                lim = math.floor(F + u)
                q = sum(o.quantity for o in asks if o.price <= lim and o.owner == 'market')
                q = min(q, -pos, self._room_side(BUY))
                if q > 0:
                    self._send(BUY, lim, q, OrderType.IOC)


# snipe800: shock sniper takes up to 800 lots (breaching the +-100 limit briefly) and IOCs the excess back
# at prices down to F -+ 6 ticks. Net of the 10k/s breach penalty. Source: strategies/marek/round2/hyp/mech.py.
PARAMS.update({"xu_on": 1, "xs_cap": 800, "xu_lim": 6.0})

# rebased on strategies/marek/round2/fillsnipe.py (mech.py + fill_room=5, xs_cap=950, xu_lim=1.0)
PARAMS.update({"fill_room": 5, "xu_on": 1, "xs_cap": 950, "xu_lim": 1.0})

# snipe_best (snipe agent, 2026-11-07): fillsnipe + robust shock oversize.
# - excess over the cap only when the filter mean is >= 4 ticks through the stale touch (re-checked at every shot,
#   so shot 2/3 can still carry it when the filter confirms a few ms later); cuts small/false/wrong-way shocks
# - ss_sticky: the sniper stays armed after fv_mm's shock window closes, but those late triggers get no excess
# - time stop: 80 ms after the trigger the excess exit IOCs widen from F -+ 1 to F -+ 4
PARAMS.update({'st_win': 0, 'st_w': 0.0, 'st_pm': None, 'st_pe': 12.0})
PARAMS.update({"xs_edge": 4.0, "xs_reval": 1, "ss_sticky": 1, "xs_nostk": 1, "xu_lim2": 4.0, "xu_ts2": 0.08})

PARAMS.update({"burst_exc": 0.2, "st_win": 1, "ps_on": 1, "ps_theta": 1.0, "ps_t0": 1.0, "ps_maxsd": 0.6})

PARAMS.update({"dr_on": 1, "dr_margin": 0.3, "dr_theta": 0.8, "dr_keep": 0.5, "dr_flat": 1})

PARAMS.update({'st_cnt': 0})
PARAMS.update({'st_cn': 0, 'st_cw': 0.02, 'st_lv': 3, 'st_tq': 0})
PARAMS.update({'st_cn': 3, 'st_cnt': 1, 'st_cw': 0.03})
# final (2026-11-07 23:44): combo + drought quoting/flatten (drought agent) + cancel-wave shock trigger (shocktime agent, st_cn/st_cnt)
# 8-tape walk-forward vs combo: r2 758,092 (+23.4k; train -3.3k, hold +26.8k), synth mean 686,206 (+10.5k)

# final2 (2026-11-07 23:55): final + take_sd_t=1.5 (turbulent take threshold 2.0 -> 1.5; regime/stack agents)
# 8-tape walk-forward vs final: r2 761,532 (+3.4k; train +6.0k, hold -2.6k), synth mean 698,861 (+12.7k), 6/7 synth up, t=3.38
PARAMS.update({"take_sd_t": 1.5})

# final3 (2026-11-08 00:06): final2 + kick=0.0 (Peter's fvs_mm update of 2026-11-07 22:01, which our lineage predates)
# vs final2: r2 763,078 (+1.5k; r2 without the 5398 window +2.0k), synth mean 713,454 (+14.6k, all 7 tapes up), t=5.5
PARAMS.update({"kick": 0.0})

# shockgen_best (shockgen agent, 2026-11-08 00:56): final3 + ps_on=0 (overfit_best) + clipped-insert sniper fixes:
# - ss_cw0=2.93: the clipped-insert pattern (bid at ask-1 / ask at bid+1, 100% direction-accurate on r2+s1..s3,
#   median ~9 ms after the jump) may trigger from onset_est+2.93 (onset estimate is often ~50 ms late)
# - ss_cw_x=1: such early triggers count as in-window for the excess
# - ss_clip_e=3: a clip trigger's first shot carries the excess when a second signal agrees (F edge >= 2 or a
#   same-side stale cancel in st_cw), instead of waiting for the filter edge >= 4 (stale side is gone by shot 2)
# vs BASE_overfit_best: r2 772.4k (ex5398 +7.8k, 5398 window 0), synth mean +2.8k (6/7 tapes up, s7 +13.0k),
# 21 shocks improved / 14 worsened; minus the single biggest synth shock (s7 5327, +11.9k) synth mean +1.1k.
PARAMS.update({"ps_on": 0, "ss_cw0": 2.93, "ss_cw_x": 1, "ss_clip_e": 3})


# ---- round 3
PARAMS.update({'lad_on': 1, 'lad_tol': 3, 'lad_deep': 2, 'lad_wait': 2010, 'lad_ss': 1, 'lad_band': 7})
# transient impact (marek/research/round3/novelty/README.md, s_dynamics/params_v3.json): a share 1 - tr_p of each
# taker's kick decays with time constant tr_T (s). y_k > 0 quotes around F - y_k * Y (fade the transient part).
PARAMS.update({'tr_on': 0, 'tr_p': 0.58, 'tr_T': 17.3, 'tr_p_v': 0.55, 'tr_T_v': 15.0, 'y_k': 0.0, 'lad_dr': 1})
PARAMS.update({'c': 0.0171, 'e': 0.61, 'lam': 0.4756, 'lam_v': 1.06, 'tr_sr': 0})  # round-3 fit (s_dynamics v3)
PARAMS.update({'eta': 0.1})  # round 3: absorbs hidden one-rung ladder orders (+60k synth, +40k r3)
# ---- round 3 selection (marek/research/round3/FINDINGS.md): each kept change is positive on round3 AND the synth mean
PARAMS.update({'eta': 0.15, 'tr_on': 1, 'tr_sr': 1, 'theta': 0.25, 'take_t': 0.3, 'pv_thr': 0.7,
               'ss_trade': 1, 'ss_trade_lv': 2})


# ---- develop angle (marek/research/round3/ideas/develop.md); all off = r3_best
PARAMS.update({'ctk_y': 0.0, 'ctk_strict': 0, 'cq_y': 0.0, 'cm_k': 0.0, 'cm_max': 1.0, 'cgk': 0.0})

# ---- bigmoves (marek/research/round3/ideas/bigmoves.md, ideas/bigmoves_trig.py). All off = identical to r3_best.
PARAMS.update({'bm_sw': 0, 'bm_sw_t0': 2.90, 'bm_sw_lv': 2, 'bm_sw_span': 3, 'bm_sw_cr': 0, 'bm_sw_edge': 10.0,
               'bm_sw_empty': 0, 'bm_sw_big_only': 0})

# ---- volatile (marek/research/round3/ideas/volatile.md, ideas/volatile_tk.py vf_*). All off = identical to r3_best.
PARAMS.update({'vf_on': 0, 'vf_th': 2.0, 'vf_keep': 1.6, 'vf_q': 100, 'vf_mom': None, 'vf_pos': 1, 'vf_shk': 0})
PARAMS.update({'cu_u': -1.0, 'cu_sd': 0.0, 'cu_t0': 3.0, 'cu_t1': 10.0, 'cu_to': 0})  # develop: calm take unload (off)
PARAMS.update({'tr_take': None, 'tr_take_sd': 2.0, 'tr_to': 0})  # calm angle tr_take (off)
PARAMS.update({'vx_u': -1.0, 'vx_age': 2.0, 'vx_mom': None, 'vx_pos': 60, 'vx_to': 0})  # develop: volatile aged exit (off)
PARAMS.update({'th_in': None})  # develop: calm theta when the market spread is >= 2 (off)
PARAMS.update({'th_red': None, 'th_red_pos': 90})  # develop: calm reducing-side theta when pinned (off)


# ======================================================================================================
# r3_dev_best settings (develop angle, 2026-11-08): round 3 853.7k (+58.7k vs r3_best), round 2 778.4k (+17.4k)
PARAMS.update({'bm_sw': 1, 'bm_sw_cr': 1, 'bm_sw_big_only': 1, 'ss_clip_e': 2})          # bigmoves C2 (shocks)
PARAMS.update({'take': 0.10, 'tr_take': 0.0, 'tr_take_sd': 1.0, 'theta_keep': 0.0})       # calm c2
PARAMS.update({'vf_on': 1, 'vf_th': 2.0, 'vf_keep': 1.5, 'vf_mom': -0.5, 'vf_pos': 60})   # volatile vfm20
PARAMS.update({'c': 0.0168, 'e': 0.616, 'jv': 0.0169, 'tr_p': 0.50, 'tr_T': 16.7,       # s_dynamics final model
               'q_cv': 0.0064, 'q_vc': 0.0166, 'sigma': 0.15})


# ======================================================================================================
# round 4 (strategies/peter/round4_findings.md): momentum forecast, coefficients = OLS of the 5 s F change on
# turbulent rows of the public tape (r4_alpha.py). mk = 0 reproduces r3_dev_best exactly.
PARAMS.update({'mg': 2.77, 'm05': -0.36, 'm1': 0.37, 'm2': 0.20, 'm5': 0.32, 'm10': -0.15, 'm_max': 20.0,
               'mk': 0.0, 'mk_calm': 0.0, 'm_thr': 0.0, 'm_cap': 100, 'm_sl': 0, 'mq': 0.0, 'm_veto': 0.0})

# ---- round-4 selection (2026-11-08): full r4 1438.3k vs dev_best 1365.9k (+72.3k, 9/12 500 s pieces), wf +64.8k,
# hold +12.8k (3/4). dr_margin: stop drought quoting at onset + 2.1 s (plateau 0.7-1.0); lamj_v / bj_v: volatile
# residual jumps refit on r4 (session db, r4_trend.py likelihood fit).
PARAMS.update({'dr_margin': 0.9, 'lamj_v': 1.0, 'bj_v': 1.2})
