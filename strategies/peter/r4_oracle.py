"""Diagnostic only (not a submission): Marek's r4_best with selected decision inputs replaced by offline truth
from the round-4 public tape, to measure how much PnL each aspect of the model could still add.

Truth (built by r4_oracle_data.py into logs/r4/oracle_truth.npz):
    S       smoothed hidden price per insert (marek/research/round4/s_dynamics, run v3: velocity plug-in)
    vol     exact volatile segments (s_dynamics/vol_segments.csv)
    onsets  shock drought onsets, jump times and sizes (flow/shock_events.csv)

Knobs (all 0 = r4_best exactly; the filter itself always keeps running, only decision inputs change):
    or_lvl   1: F := S(now)                             (perfect level)
    or_mov   1: F += S(now + or_h) - S(now)             (perfect next move over or_h s, on top of F)
             2: same but using the filter's own F only outside shock windows / only inside (or_mov_w)
    or_sd    sd fed to the decisions while or_lvl is on (default 0.05)
    or_reg   1: _turbulent := true volatile segment; 2: also the filter's regime prob pv := truth
    or_on    1: drought onset estimate := true onset
    or_jmp   1: from the true onset to the jump, F += the coming jump J (perfect shock direction and size)
    or_win   restrict or_lvl / or_mov to 'all' | 'calm' | 'vol' | 'noshock' (true labels)
"""
import importlib.util
import math
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
_spec = importlib.util.spec_from_file_location('r4_best_oracle_base', _ROOT / 'strategies/marek/round4/r4_best.py')
M = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M)
PARAMS = M.PARAMS
PARAMS.update({'or_lvl': 0, 'or_mov': 0, 'or_h': 1.0, 'or_sd': 0.05, 'or_reg': 0, 'or_on': 0, 'or_jmp': 0,
               'or_win': 'all', 'or_truth': str(_ROOT / 'logs/r4/oracle_truth.npz')})

_T = {}


def _truth(path):
    if path not in _T:
        z = np.load(path)
        _T[path] = {k: z[k] for k in z.files}
    return _T[path]


class Oracle(M.FVAShock):
    def __init__(self):
        super().__init__()
        self.or_saved = None

    # ---- truth lookups
    def _tr(self):
        return _truth(self.p['or_truth'])

    def _S(self, t):
        z = self._tr()
        i = int(np.searchsorted(z['t'], t, 'right'))  # first insert after t: S includes every event up to t
        return float(z['S'][min(i, len(z['S']) - 1)])

    def _vol(self, t):
        z = self._tr()
        j = int(np.searchsorted(z['va'], t, 'right')) - 1
        return j >= 0 and t < z['vb'][j]

    def _shockwin(self, t):
        z = self._tr()
        j = int(np.searchsorted(z['on'], t + 1.0, 'right')) - 1
        return j >= 0 and z['on'][j] - 1.0 <= t < z['on'][j] + 20.0

    def _win_ok(self, t):
        w = self.p['or_win']
        if w == 'all':
            return True
        sh = self._shockwin(t)
        if w == 'noshock':
            return not sh
        if sh:
            return False
        return self._vol(t) == (w == 'vol')

    # ---- injection
    def on_order_book_change(self, snapshot, trades):
        f = self.f
        if f is not None and self.or_saved is not None:
            f.mean, f.sd = self.or_saved
            self.or_saved = None
        if f is not None and self.p['or_reg'] == 2:
            f.pv = 1 - 1e-4 if self._vol(snapshot.timestamp_us * 1e-6) else 1e-4
        super().on_order_book_change(snapshot, trades)
        f = self.f
        if f is not None and self.or_saved is not None:  # sniper code ran with oracle F; restore for the filter
            f.mean, f.sd = self.or_saved
            self.or_saved = None

    def _quote(self, f, bb, ba, bids, asks):
        p = self.p
        t = self.now_t
        F, sd = f.mean, f.sd
        new_F, new_sd = F, sd
        if (p['or_lvl'] or p['or_mov']) and self._win_ok(t):
            if p['or_lvl']:
                new_F, new_sd = self._S(t), p['or_sd']
            if p['or_mov']:
                new_F += self._S(t + p['or_h']) - self._S(t)
        if p['or_jmp']:
            z = self._tr()
            j = int(np.searchsorted(z['on'], t, 'right')) - 1
            if j >= 0 and z['on'][j] <= t < z['tj'][j]:
                new_F += float(z['J'][j])
        if new_F != F or new_sd != sd:
            self.or_saved = (F, sd)
            f.mean, f.sd = new_F, new_sd
        super()._quote(f, bb, ba, bids, asks)

    def _turbulent(self, f):
        if self.p['or_reg']:
            return self._vol(self.now_t)
        return super()._turbulent(f)

    def _onset(self, sg):
        if self.p['or_on']:
            z = self._tr()
            j = int(np.searchsorted(z['on'], sg, 'right')) - 1
            return float(z['on'][j]) if j >= 0 and sg - z['on'][j] < 3.5 else super()._onset(sg)
        return super()._onset(sg)
