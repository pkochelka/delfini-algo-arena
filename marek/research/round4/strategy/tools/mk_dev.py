"""Build strategies/marek/round4/r4_dev.py from strategies/marek/round3/r3_hmm.py + round-4 knob sets (all off)."""
import pathlib
ROOT = pathlib.Path(__file__).resolve().parents[5]
src = (ROOT / 'strategies/marek/round3/r3_hmm.py').read_text()

def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (old, src.count(old))
    src = src.replace(old, new)

rep('"""r3_hmm: r3_dev_best', '"""r4_dev: r3_hmm + round-4 knob sets (marek/research/round4/hypothses.md), all off = r3_hmm:\n'
    '  vel_*  H2 velocity term: F_eff = F + k * (F - EMA_tau(F)) [+ g * clip(F - mid)], calm / turbulent gains\n'
    '  vd_*   H2b drift in the filter predict step (S moves by v dt between observations)\n'
    '  ts_*   taker side law P(buy) = Phi((S - mid) / 4.42) as a likelihood on the S grid (TWAP children excluded)\n'
    '  tw_*   H1/H6 online TWAP detector (IOC every 500,000 us, same side): F_eff lean, quote pull\n'
    '  j15_*  H1 15-lot joiner (insert 1 us after a fill/cancel) dropped from the maker likelihood\n'
    'Base r3_hmm: r3_dev_best')

# ---- FV: EMA of the mean, drift in predict, taker side likelihood
rep("        self.pv = 0.0  # P(volatile)\n",
    "        self.pv = 0.0  # P(volatile)\n        self.ema = float(s0)  # round 4: EMA of the mean (velocity = (mean - ema) / tau)\n"
    "        self.vdr = 0.0  # round 4: drift (ticks / s) applied in predict\n")
rep("""        if p['tr_on'] and self.Y:
            T = (1 - pv) * p['tr_T'] + pv * p['tr_T_v']""",
"""        if p['vel_tau'] > 0:
            self.ema += -math.expm1(-dt / p['vel_tau']) * (self.mean - self.ema)
        if self.vdr:
            self.shift(self.vdr * dt)
        if p['tr_on'] and self.Y:
            T = (1 - pv) * p['tr_T'] + pv * p['tr_T_v']""")
rep("""    def taker(self, t: float, is_buy: bool, price, filled: int) -> None:
        self.predict(t)
        p = self.p
""", """    def side_obs(self, is_buy: bool, mid: float, slope: float) -> None:
        \"\"\"Round 4 side law: P(buy | S) = logistic(slope * (S - mid)) (flow/sidelaw.py), multiplied into the grid.\"\"\"
        S = self.c * self.h + self.r + self.k
        z = slope * (S - mid)
        L = 1.0 / (1.0 + np.exp(-z if is_buy else z))
        post = self.w * L
        self._post(post, float(post.sum()), False)

    def taker(self, t: float, is_buy: bool, price, filled: int) -> None:
        self.predict(t)
        p = self.p
""")

# ---- strategy state
rep("        self.lq_ids: dict = {}  # hmm angle: limit-price quotes, id -> side\n",
    "        self.lq_ids: dict = {}  # hmm angle: limit-price quotes, id -> side\n"
    "        self.r4_jumps = 0  # round 4: jump count seen by the velocity reset\n"
    "        self.tw_last = None  # round 4 TWAP: (t_us, sign) of the last 500 ms-linked taker candidate\n"
    "        self.tw_n = 0  # children seen in the current chain (>= 2 = TWAP episode active)\n"
    "        self.tw_s = 0  # side of the active episode (+1 buy, -1 sell)\n"
    "        self.tw_t0 = 0.0\n"
    "        self.prev_us = 0  # timestamp of the previous public event\n"
    "        self.prev_kind = 0  # 1 = it had a trade, 2 = it removed volume (cancel) on some side\n"
    "        self.n_j15 = 0\n"
    "        self.n_tw = 0\n")

# ---- event loop: taker side law, TWAP detection, 15-lot joiner
rep("""            rested = (newb is not None and newb.order_id == tid) or (newa is not None and newa.order_id == tid)
            if not rested:  # a crossing limit order (only possible against us) is no IOC: no impact
                f.taker(t, sgn > 0, None, q)""",
"""            rested = (newb is not None and newb.order_id == tid) or (newa is not None and newa.order_id == tid)
            if not rested:  # a crossing limit order (only possible against us) is no IOC: no impact
                is_tw = self._tw_taker(snapshot.timestamp_us, sgn, q) if p['tw_on'] else False
                if p['ts_on'] and pbb is not None and pba is not None and not (is_tw and p['ts_tw']) \\
                        and self.sg_t is None:
                    f.predict(t)
                    f.side_obs(sgn > 0, 0.5 * (pbb + pba), p['ts_slope'])
                f.taker(t, sgn > 0, None, q)""")
rep("""        elif newb is not None:
            if lad:""", """        elif p['j15_on'] and (newb is not None or newa is not None) and self._j15(snapshot.timestamp_us, newb, newa, pbb, pba):
            pass
        elif newb is not None:
            if lad:""")
# record previous event kind at the end of the public-feed processing (before quoting)
rep("""        if not bids or not asks or snapshot.timestamp_us < p['trade_start_us']:
            return
        self._quote(f, bids[0].price, asks[0].price, bids, asks)""",
"""        self.prev_us = snapshot.timestamp_us
        self.prev_kind = 1 if q else (0 if (newb is not None or newa is not None) else 2)
        if p['tw_on'] and self.tw_n and snapshot.timestamp_us - self.tw_last[0] > 500000 + p['tw_miss_us']:
            self.tw_n = 0  # a missed 500 ms slot: the episode is over
            self.tw_end = t
        if p['vel_tau'] > 0:
            if f.jumps != self.r4_jumps or self.sg_t is not None:
                self.r4_jumps = f.jumps
                f.ema = f.mean
            if p['vd_k']:
                g = f.mean - f.ema
                k = p['vd_k'] * f.pv + p['vd_kc'] * (1 - f.pv)
                f.vdr = k * max(-p['vel_max'], min(p['vel_max'], g)) / p['vel_tau'] if self.sg_t is None else 0.0
        if not bids or not asks or snapshot.timestamp_us < p['trade_start_us']:
            return
        self._quote(f, bids[0].price, asks[0].price, bids, asks)

    # ---------------------------------------------------------------- round 4 agents
    def _tw_taker(self, t_us: int, sgn: int, q: int) -> bool:
        \"\"\"Online TWAP detector: a market IOC exactly 500,000 us (+- tw_tol) after the previous candidate, same side.
        A candidate is a fill of exactly tw_q lots (or any taker in the slot of an active chain).\"\"\"
        p = self.p
        last = self.tw_last
        if self.tw_n and abs(t_us - last[0] - 500000) <= p['tw_tol'] and sgn == last[1]:
            self.tw_last = (t_us, sgn)
            self.tw_n += 1
            if self.tw_n == 2:
                self.tw_s = sgn
                self.tw_t0 = self.now_t
                self.n_tw += 1
            return True
        if q == p['tw_q'] and self.tw_n == 0:
            self.tw_last = (t_us, sgn)
            self.tw_n = 1
        return False

    def _j15(self, t_us, newb, newa, pbb, pba) -> bool:
        \"\"\"Round 4 15-lot joiner: a 15 (or 10 = remainder) lot insert at the same-side touch exactly 1 us after a
        fill or cancel row. True = drop it from the maker likelihood.\"\"\"
        v = newb if newb is not None else newa
        if t_us - self.prev_us != 1 or self.prev_kind == 0 or v.quantity not in self.p['j15_q']:
            return False
        own = pbb if newb is not None else pba
        if own is not None and v.price != own and not (newb is not None and v.price > own) \\
                and not (newa is not None and v.price < own):
            return False
        self.n_j15 += 1
        return True""")

# ---- F_eff in _quote
rep("""        F, sd = f.mean - p['y_k'] * f.Y, f.sd
        turb = self._turbulent(f)""",
"""        F, sd = f.mean - p['y_k'] * f.Y, f.sd
        turb = self._turbulent(f)
        r4d = self._r4_shift(f, turb)
        if not p['r4_veto']:
            F += r4d""")
rep("""            tk = take + take_sd * sd
            lo_b, hi_a = F - tk, F + tk
""", """            tk = take + take_sd * sd
            lo_b, hi_a = F - tk, F + tk
        if p['r4_veto']:  # round 4: the lean only makes takes against it harder
            lo_b += min(r4d, 0.0)
            hi_a += max(r4d, 0.0)
""")
rep("""    def _lq(self, F: float, turb: bool,""", """    def _r4_shift(self, f, turb: bool) -> float:
        \"\"\"Round 4 fair-value lean: velocity (H2), book-lag (S - mid) momentum, TWAP push (H6b).\"\"\"
        p = self.p
        d = 0.0
        if self.sg_t is not None:
            return 0.0
        if p['vel_tau'] > 0:
            k = p['vel_kv'] if turb else p['vel_kc']
            if k:
                g = f.mean - f.ema
                d += k * max(-p['vel_max'], min(p['vel_max'], g))
        gk = p['gm_kv'] if turb else p['gm_kc']
        if gk and self.mbb is not None and self.mba is not None:
            g = f.mean - 0.5 * (self.mbb + self.mba)
            d += gk * max(-p['gm_max'], min(p['gm_max'], g))
        if p['tw_on'] and self.tw_n >= 2 and p['tw_k']:
            d += p['tw_k'] * self.tw_s
        return d

    def _lq(self, F: float, turb: bool,""")

rep("""        kp_b, kp_a = keep + wide + sk, keep + wide - sk
""", """        kp_b, kp_a = keep + wide + sk, keep + wide - sk
        if p['tw_th'] and p['tw_on'] and self.tw_n >= 2 and not turb:  # H6a: back off the side the TWAP hits
            if self.tw_s > 0:
                th_a += p['tw_th']
                kp_a += p['tw_th']
            else:
                th_b += p['tw_th']
                kp_b += p['tw_th']
""")

# ---- H2c: Kalman velocity state alongside the grid (v, Pv, Psv; OU prior per regime, pv-mixed)
rep("""        self.vdr = 0.0  # round 4: drift (ticks / s) applied in predict
""", """        self.vdr = 0.0  # round 4: drift (ticks / s) applied in predict
        self.kv = 0.0  # round 4 H2c: velocity estimate (ticks / s), its variance and covariance with S
        self.kPv = 0.0
        self.kPsv = 0.0
""")
rep("""        if self.vdr:
            self.shift(self.vdr * dt)
""", """        if self.vdr:
            self.shift(self.vdr * dt)
        if p['kv_on']:
            tau = (1 - pv) * p['kv_tau_c'] + pv * p['kv_tau_v']
            s2 = (1 - pv) * p['kv_sd_c'] ** 2 + pv * p['kv_sd_v'] ** 2
            if self.kv and not self.drought:
                self.shift(self.kv * dt)
            ph = math.exp(-dt / tau)
            self.kPsv = ph * (self.kPsv + self.kPv * dt)
            self.kPv = ph * ph * self.kPv + s2 * (1 - ph * ph)
            self.kv *= ph
""")
rep("""    def _post(self, post: np.ndarray, z: float, count_ll: bool) -> None:
""", """    def _post(self, post: np.ndarray, z: float, count_ll: bool) -> None:
        P0 = 0.0
        if self.p['kv_on'] and self.kPsv:
            w0 = self.w
            m0 = float(w0 @ self.k) / float(w0.sum())
            P0 = float(w0 @ self.k2) / float(w0.sum()) - m0 * m0
""")
rep("""        if abs(self.mean - old) > self.p['jump_det'] and old_sd < self.p['jump_sd']:
            self._jumped()
""", """        if self.p['kv_on'] and self.kPsv and P0 > 1e-6 and not self.drought:
            P1 = self.sd * self.sd
            g = self.kPsv / P0
            K = max(0.0, 1.0 - P1 / P0)
            self.kv += self.p['kv_g'] * g * (self.mean - old)
            cap = self.p['kv_cap']
            self.kv = max(-cap, min(cap, self.kv))
            self.kPv = max(self.kPv - K * self.kPsv * g, 1e-6)
            self.kPsv *= 1.0 - K
        if abs(self.mean - old) > self.p['jump_det'] and old_sd < self.p['jump_sd']:
            self._jumped()
""")
rep("""    def _jumped(self) -> None:
        self.exc = min(self.exc + self.p['kick'], self.p['exc_max'])""", """    def _jumped(self) -> None:
        self.kv = self.kPsv = 0.0
        self.exc = min(self.exc + self.p['kick'], self.p['exc_max'])""")
src += """

# ======================================================================================================
# ---- round 4 knob sets (marek/research/round4/strategy/README.md). All off = r3_hmm.
PARAMS.update({'vel_tau': 0.0, 'vel_kc': 0.0, 'vel_kv': 0.0, 'vel_max': 10.0,  # H2: lean k * (F - EMA_tau F)
               'vd_k': 0.0, 'vd_kc': 0.0,  # H2b: filter drift v = vd_k * (F - EMA) / tau (pv-weighted calm vd_kc)
               'gm_kc': 0.0, 'gm_kv': 0.0, 'gm_max': 3.0,  # book-lag momentum lean k * clip(F - mid)
               'ts_on': 0, 'ts_slope': 0.37, 'ts_tw': 1,  # taker side law likelihood (ts_tw: skip TWAP children)
               'tw_on': 0, 'tw_q': 160, 'tw_tol': 3, 'tw_miss_us': 2000, 'tw_k': 0.0, 'tw_th': 0.0,  # TWAP detector / lean
               'j15_on': 0, 'j15_q': (15, 10),
               'r4_veto': 0,
               'kv_on': 0, 'kv_tau_c': 0.5, 'kv_tau_v': 2.1, 'kv_sd_c': 0.3, 'kv_sd_v': 3.0, 'kv_g': 1.0, 'kv_cap': 10.0})  # 1: the lean (vel/gm/tw) only vetoes takes against it, F unchanged  # 15-lot joiner filter
"""
(ROOT / 'strategies/marek/round4/r4_dev.py').write_text(src)
print('ok')
