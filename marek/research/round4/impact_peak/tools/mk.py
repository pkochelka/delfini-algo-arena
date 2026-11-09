"""Build strategies/marek/round4/r4_impactpeak.py from r4_best.py + impact-peak knobs (all off = r4_best)."""
import pathlib
ROOT = pathlib.Path(__file__).resolve().parents[5]
src = (ROOT / 'strategies/marek/round4/r4_best.py').read_text()
def rep(a, b, n=1):
    global src
    assert src.count(a) == n, (a, src.count(a)); src = src.replace(a, b)
rep('"""r4_best: r4_dev with the round-4 filter refit on (settings at the end of the file).',
    '"""r4_impactpeak: r4_best + impact-peak knobs (marek/research/round4/impact_peak/README.md), all off = r4_best.\n'
    '  ip_*  decayed signed taker-kick flows Xf (tau ip_tf) / Xs (tau ip_ts); lean L = kf Xf + ks Xs - yk Y per regime,\n'
    '        applied to F for takes and passive quotes (ip_mode 0), passive only (1) or takes only (2)\n'
    '  fp_*  fade-after-peak: volatile flatten quote unblocked when old flow (Y) backs the position and new flow (Xf) has died\n'
    '  vx_y  volatile aged exit (vx_*) gated on Y backing the position and Xf dead\n\n'
    'r4_best: r4_dev with the round-4 filter refit on (settings at the end of the file).')
rep("        self.n_j15 = 0\n        self.n_tw = 0\n",
    "        self.n_j15 = 0\n        self.n_tw = 0\n        self.ip_xf = self.ip_xs = 0.0  # impact peak: decayed signed kick flows\n        self.ip_t = 0.0\n")
rep("                f.taker(t, sgn > 0, None, q)\n",
    "                f.taker(t, sgn > 0, None, q)\n"
    "                if p['ip_on']:\n"
    "                    self._ip_decay(t)\n"
    "                    pv_ = f.pv\n"
    "                    d_ = (1 - pv_) * p['c'] * q ** p['e'] + pv_ * p['c_v'] * q ** p['e_v']\n"
    "                    self.ip_xf += d_ if sgn > 0 else -d_\n"
    "                    self.ip_xs += d_ if sgn > 0 else -d_\n")
rep("    # ---------------------------------------------------------------- round 4 agents\n",
    "    def _ip_decay(self, t: float) -> None:\n"
    "        dt = t - self.ip_t\n"
    "        if dt > 0:\n"
    "            self.ip_t = t\n"
    "            self.ip_xf *= math.exp(-dt / self.p['ip_tf'])\n"
    "            self.ip_xs *= math.exp(-dt / self.p['ip_ts'])\n\n"
    "    def _ip_lean(self, f, turb: bool) -> float:\n"
    "        p = self.p\n"
    "        if not p['ip_on'] or self.sg_t is not None:\n"
    "            return 0.0\n"
    "        self._ip_decay(self.now_t)\n"
    "        if turb:\n"
    "            L = p['ip_kfv'] * self.ip_xf + p['ip_ksv'] * self.ip_xs - p['ip_yv'] * f.Y\n"
    "        else:\n"
    "            L = p['ip_kfc'] * self.ip_xf + p['ip_ksc'] * self.ip_xs - p['ip_yc'] * f.Y\n"
    "        m = p['ip_max']\n"
    "        return max(-m, min(m, L))\n\n"
    "    # ---------------------------------------------------------------- round 4 agents\n")
rep("        self.turb = turb\n        if turb:\n            cap, theta",
    "        self.turb = turb\n"
    "        ipL = self._ip_lean(f, turb) if p['ip_on'] else 0.0\n"
    "        if ipL and p['ip_mode'] != 1:\n"
    "            F += ipL\n"
    "        if turb:\n            cap, theta")
rep("        # ---- passive quotes\n",
    "        # ---- passive quotes\n"
    "        if ipL and p['ip_mode'] == 1:\n"
    "            F += ipL\n"
    "        elif ipL and p['ip_mode'] == 2:\n"
    "            F -= ipL\n")
rep("            pos_gate = (F - 0.5 * (self.mbb + self.mba)) * (1 if pos > 0 else -1) > p['vf_mom']\n",
    "            pos_gate = (F - 0.5 * (self.mbb + self.mba)) * (1 if pos > 0 else -1) > p['vf_mom']\n"
    "        fp = False\n"
    "        if vf and p['fp_on'] and pos and self.sg_t is None:\n"
    "            self._ip_decay(self.now_t)\n"
    "            sp_ = 1 if pos > 0 else -1\n"
    "            fp = sp_ * f.Y >= p['fp_y'] and sp_ * self.ip_xf <= p['fp_xf']\n"
    "            if fp:\n"
    "                pos_gate = False\n")
rep("        if vf:\n            passive = True\n            th_b = th_a = p['vf_th']\n            kp_b = kp_a = p['vf_keep']\n",
    "        if vf:\n            passive = True\n            th_b = th_a = p['vf_th'] if not fp else p['fp_th']\n"
    "            kp_b = kp_a = p['vf_keep'] if not fp else p['fp_keep']\n")
rep("            if age >= p['vx_age'] and mom_ok:\n",
    "            if p['vx_y'] is not None:\n"
    "                self._ip_decay(self.now_t)\n"
    "                sp_ = 1 if lng else -1\n"
    "                mom_ok = mom_ok and sp_ * f.Y >= p['vx_y'] and sp_ * self.ip_xf <= p['vx_xf']\n"
    "            if age >= p['vx_age'] and mom_ok:\n")
src += ("\n\n# ======================================================================================================\n"
        "# impact peak knobs (marek/research/round4/impact_peak/README.md). All off = r4_best.\n"
        "PARAMS.update({'ip_on': 0, 'ip_tf': 1.5, 'ip_ts': 12.0, 'ip_kfv': 0.0, 'ip_ksv': 0.0, 'ip_yv': 0.0,\n"
        "               'ip_kfc': 0.0, 'ip_ksc': 0.0, 'ip_yc': 0.0, 'ip_max': 6.0, 'ip_mode': 0,\n"
        "               'fp_on': 0, 'fp_y': 3.0, 'fp_xf': 1.0, 'fp_th': 2.0, 'fp_keep': 1.5,\n"
        "               'vx_y': None, 'vx_xf': 1e9})\n")
(ROOT / 'strategies/marek/round4/r4_impactpeak.py').write_text(src)
print('ok')
