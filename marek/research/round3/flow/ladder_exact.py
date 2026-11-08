"""Exact 2 ms chains: link insert i -> any later insert j with t_j - t_i in [1.990, 2.010] ms. Tabulate side/price/size relation."""
import numpy as np, pandas as pd
from pathlib import Path
D = Path(__file__).parent; C = D / 'cache'
r = pd.read_pickle(C / 'rows.pkl'); ins = r[r.op == 0].reset_index(drop=True)
t = ins.t.to_numpy(); us = np.rint(t * 1e6).astype(np.int64)
lo = np.searchsorted(us, us + 1990); hi = np.searchsorted(us, us + 2010)
cnt = hi - lo
print('number of inserts in [1.99,2.01] ms after each insert:', np.bincount(cnt))
# exact-offset histogram in us for those
i = np.flatnonzero(cnt > 0); j = lo[i]
d = us[j] - us[i]; print('offset us counts:', dict(zip(*np.unique(d, return_counts=True))))
# background: in [2.5,2.52] ms window
lo2 = np.searchsorted(us, us + 2490); hi2 = np.searchsorted(us, us + 2510); print('background count in 20us window', (hi2 - lo2).sum(), 'vs', cnt.sum())
ioc = ins.ioc.to_numpy(); s = ins.side.to_numpy(); p = ins.px.to_numpy(); v = ins.vol.to_numpy()
ex = i[(d >= 1999) & (d <= 2001)]; jx = lo[ex]
print('pairs exact 2ms', len(ex), 'ioc i/j', ioc[ex].mean(), ioc[jx].mean(), 'same side', (s[ex] == s[jx]).mean())
dp = (p[jx] - p[ex]) * np.where(s[ex] == 0, -1, 1)
print('price step away from touch:', dict(zip(*np.unique(dp, return_counts=True))))
np.save(C / 'pairs2ms.npy', np.c_[ex, jx])
