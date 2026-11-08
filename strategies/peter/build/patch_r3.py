"""Rebuild r3_mm.py = fvs_mm.py + round-3 patches (single-file submission)."""
p_src, p_dst = 'strategies/peter/fvs_mm.py', 'strategies/peter/r3_mm.py'
s = open(p_src, encoding='utf8').read()
def rep(a, b):
    global s
    assert s.count(a) == 1, (a[:60], s.count(a)); s = s.replace(a, b)
rep("""    'taker_obs': 0,
}""", """    'taker_obs': 0,
    # ---- round 3 (round3_findings.md). Deep-ladder maker: ~5.2 inserts/s at 4-8 ticks from S, log-size
    # N(4.67, 1.08) against the regular makers' N(2.7, 1.7) at 194/s. An insert is a ladder order with posterior
    # p_lad(size); its likelihood is then flat over S - price in [lad_lo, lad_hi].
    'lad': 1,
    'lad_rate': 5.2 / 194,
    'lad_mu': 4.67, 'lad_s': 1.08, 'reg_mu': 2.7, 'reg_s': 1.7,
    'lad_lo': 3.5, 'lad_hi': 8.6,
    # transient impact: S = L + X, X decays with tr_tau; a share of each taker's impact goes to X
    # (calm tr_c, volatile tr_v); phi_v / phi_c: share of the filter's other mean moves (jumps) attributed to X
    'tr_c': 0.0, 'tr_v': 0.0, 'tr_tau': 20.0, 'phi_v': 0.0, 'phi_c': 0.0,
    'tr_filter': 1,  # 1: the filter's prior drifts S back as X decays
    'fwd_t': 0.0,  # s: takes price against E[S] fwd_t s ahead (the X decay), quotes against fwd_q s ahead
    'fwd_q': 0.0,
}""")
rep("""        self.kern: dict[int, np.ndarray] = {}
""", """        self.kern: dict[int, np.ndarray] = {}
        self.X = 0.0  # transient part of S
        if p['lad']:
            lo, hi = p['lad_lo'], p['lad_hi']
            Ll = ((x >= lo) & (x < hi)) / (hi - lo) + 1e-4
            self.T_l = (Ll, None, Ll[::-1].copy(), None)
            self.lad_cache: dict[int, float] = {}
""")
rep("""        self.t = t
        p = self.p
        w = self.w
        pv = self.pv
""", """        self.t = t
        p = self.p
        if self.X:
            dx = self.X * -math.expm1(-dt / p['tr_tau'])
            self.X -= dx
            if p['tr_filter']:
                self.shift(-dx)
            if abs(self.X) < 1e-4:
                self.X = 0.0
        w = self.w
        pv = self.pv
""")
rep("""        if abs(m) > 3.0:
            self._recentre(round(m * self.inv_h))""", """        if abs(m) > 3.0:
            self._recentre(round(m * self.inv_h))
        ph = self.pv * self.p['phi_v'] + (1 - self.pv) * self.p['phi_c']
        if ph and not self.hot and not self.drought:
            self.X += ph * (self.mean - old)""")
rep("""    def _observe(self, is_bid: bool, price: int, clamped: bool, count_ll: bool) -> None:
        w = self.w
        if self.drought or not self.two:""", """    def p_lad(self, qty: int) -> float:
        \"\"\"Posterior probability that a market insert of this size is a deep-ladder order.\"\"\"
        v = self.lad_cache.get(qty)
        if v is None:
            p = self.p
            z = math.log(max(qty, 1))
            fl = math.exp(-0.5 * ((z - p['lad_mu']) / p['lad_s']) ** 2) / p['lad_s']
            fr = math.exp(-0.5 * ((z - p['reg_mu']) / p['reg_s']) ** 2) / p['reg_s']
            r = p['lad_rate']
            v = self.lad_cache[qty] = r * fl / (r * fl + (1 - r) * fr)
        return v

    def _mix_like(self, is_bid, price, clamped):
        \"\"\"Regime-mixed regular-maker likelihood (updates P(volatile) like _observe).\"\"\"
        Lc = self._slice(self.T_c, is_bid, price, clamped)
        if not self.two or Lc is None:
            return Lc
        Lv = self._slice(self.T_v, is_bid, price, clamped)
        w = self.w
        zc = float(w @ Lc)
        zv = float(w @ Lv)
        pv = self.pv
        z = (1 - pv) * zc + pv * zv
        if z > 1e-300:
            self.pv = min(max(pv * zv / z, 1e-4), 1 - 1e-4)
        return (1 - pv) * Lc + pv * Lv

    def _observe(self, is_bid: bool, price: int, clamped: bool, count_ll: bool, qty: int = 0) -> None:
        w = self.w
        if qty and self.p['lad'] and not clamped and not self.drought:
            pl = self.p_lad(qty)
            if pl > 0.01:
                Lr = self._mix_like(is_bid, price, clamped)
                if Lr is None:
                    return
                L = (1 - pl) * Lr + pl * self._slice(self.T_l, is_bid, price, False)
                post = w * L
                self._post(post, float(post.sum()), count_ll)
                return
        if self.drought or not self.two:""")
rep("""    def maker(self, t: float, is_bid: bool, price: int, opp: int | None, count_ll: bool = True) -> None:""",
    """    def maker(self, t: float, is_bid: bool, price: int, opp: int | None, count_ll: bool = True, qty: int = 0) -> None:""")
rep("""        clamped = opp is not None and (price == opp - 1 if is_bid else price == opp + 1)
        if not clamped:""", """        clamped = opp is not None and (price == opp - 1 if is_bid else price == opp + 1)
        if not clamped and qty and self.p['lad'] and not self.drought and self.p_lad(qty) > 0.5:
            dd = (self.mean - price) if is_bid else (price - self.mean)
            if dd > self.K * self.h - 2:  # a deep ladder order far away: no restart evidence
                return
        if not clamped:""")
rep("""        self._observe(is_bid, price, clamped, count_ll)

    def taker""", """        self._observe(is_bid, price, clamped, count_ll, qty)

    def taker""")
rep("""            d = (1 - pv) * p['c'] * filled ** p['e'] + pv * p['c_v'] * filled ** p['e_v']
            self.shift(d if is_buy else -d)""", """            dc = p['c'] * filled ** p['e']
            dv = p['c_v'] * filled ** p['e_v']
            d = (1 - pv) * dc + pv * dv
            self.shift(d if is_buy else -d)
            x = (1 - pv) * p['tr_c'] * dc + pv * p['tr_v'] * dv
            self.X += x if is_buy else -x""")
rep("f.maker(t, True, newb.price, pba)", "f.maker(t, True, newb.price, pba, qty=newb.quantity)")
rep("f.maker(t, False, newa.price, pbb)", "f.maker(t, False, newa.price, pbb, qty=newa.quantity)")
rep("""        p = self.p
        F, sd = f.mean, f.sd
        turb = self._turbulent(f)""", """        p = self.p
        F0, sd = f.mean, f.sd
        X = f.X
        Ft = F0 - X * -math.expm1(-p['fwd_t'] / p['tr_tau']) if X and p['fwd_t'] else F0
        F = F0 - X * -math.expm1(-p['fwd_q'] / p['tr_tau']) if X and p['fwd_q'] else F0
        turb = self._turbulent(f)""")
rep("""            tk = take + take_sd * sd
            lo_b, hi_a = F - tk, F + tk""", """            tk = take + take_sd * sd
            lo_b, hi_a = Ft - tk, Ft + tk""")
open(p_dst, 'w', encoding='utf8').write(s)
