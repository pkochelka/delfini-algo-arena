"""Look for multi-order batches of the new maker participant: consecutive maker inserts within dt."""
import numpy as np, pandas as pd
from pathlib import Path
D = Path(__file__).parent; C = D / 'cache'
r = pd.read_pickle(C / 'rows.pkl'); ins = r[r.op == 0].reset_index(drop=True)
reg = (np.load(C / 'hmm_joint_2.npy')[:, 1] > .5)
mk = ins[~ins.ioc].reset_index(drop=True)
t = mk.t.to_numpy(); v = mk.vol.to_numpy(); s = mk.side.to_numpy(); p = mk.px.to_numpy()
R = reg[t.astype(int)]
dt = np.diff(t) * 1e3
for nm, sel_r, lo, hi in [('vol M', True, 20, 70), ('calm L', False, 90, 320), ('vol S', True, 1, 10), ('calm M', False, 20, 70)]:
    a = (R[:-1] == sel_r) & (v[:-1] >= lo) & (v[:-1] <= hi)
    for lim in [2, 5, 20]:
        close = a & (dt < lim)
        nb = (v[1:] >= lo) & (v[1:] <= hi)
        print(f'{nm} next<{lim}ms: n {close.sum()} frac {close.mean() / a.mean():.3f} next-in-band {nb[close].mean():.3f} (base {nb[a].mean():.3f}) '
              f'same side {(s[1:] == s[:-1])[close & nb].mean():.3f} same size {(v[1:] == v[:-1])[close & nb].mean():.3f} '
              f'|dp| {np.round(np.bincount(np.minimum(np.abs(p[1:] - p[:-1])[close & nb], 9).astype(int), minlength=10) / max((close & nb).sum(), 1), 2)}')
