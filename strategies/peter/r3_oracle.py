"""Diagnostic only (not a submission): r3_man.py with its decisions fed the generator's truth on a synthetic tape.

    or_path   data/synth/r3_v1_seed<k>.spath.npy (r3_synth_s.py): F := true S (or_mode 1) or its permanent part Fp
              (or_mode 2), or_lag s old; sd := or_sd. The filter itself keeps running; only the decision inputs change.
    or_reg    1: the turbulent switch := the true volatile regime (from the tape's .truth.json)
Headroom = oracle PnL - r3_man PnL on the same tape.
"""
import importlib.util, json
from pathlib import Path
import numpy as np

_spec = importlib.util.spec_from_file_location('r3_man_oracle_base', Path(__file__).resolve().parent / 'r3_man.py')
M = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M)
PARAMS = M.PARAMS
PARAMS.update({'or_path': '', 'or_mode': 1, 'or_sd': 0.05, 'or_lag': 0.0, 'or_reg': 0})


class Oracle(M.FVAShock):
    def __init__(self):
        super().__init__()
        self.sp = None
        self.segs = None

    def _load(self):
        p = self.p
        a = np.load(p['or_path'])
        self.sp_t, self.sp_S, self.sp_F = a[:, 0].copy(), a[:, 1].copy(), a[:, 2].copy()
        tj = Path(p['or_path'].replace('.spath.npy', '.truth.json'))
        tr = json.loads(tj.read_text())
        self.seg_a = np.array([s[0] for s in tr['segs']]); self.seg_v = [s[2] for s in tr['segs']]
        self.sp = True

    def _truth(self):
        p = self.p
        if self.sp is None:
            self._load()
        i = int(np.searchsorted(self.sp_t, self.now_t - p['or_lag'], 'right')) - 1
        i = max(i, 0)
        return self.sp_S[i] if p['or_mode'] == 1 else self.sp_F[i]

    def _turbulent(self, f):
        if self.p['or_reg']:
            if self.sp is None:
                self._load()
            j = int(np.searchsorted(self.seg_a, self.now_t, 'right')) - 1
            return bool(self.seg_v[max(j, 0)])
        return super()._turbulent(f)

    def _quote(self, f, bb, ba, bids, asks):
        if self.p['or_path'] and self.p['or_mode']:
            f.mean = float(self._truth())
            f.sd = self.p['or_sd']
        super()._quote(f, bb, ba, bids, asks)
