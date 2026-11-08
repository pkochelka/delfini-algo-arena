"""For each maker insert, find the next maker insert on the same side at price one tick further from the touch
(bid-1 / ask+1). Distribution of the time to it, by regime/size band, vs the background (long gaps)."""
import numpy as np, pandas as pd
from pathlib import Path
D = Path(__file__).parent; C = D / 'cache'
r = pd.read_pickle(C / 'rows.pkl'); ins = r[r.op == 0].reset_index(drop=True)
reg = (np.load(C / 'hmm_joint_2.npy')[:, 1] > .5)
mk = ins[~ins.ioc].reset_index(drop=True)
t = mk.t.to_numpy(); v = mk.vol.to_numpy(); s = mk.side.to_numpy(); p = mk.px.to_numpy(); n = len(t)
R = reg[t.astype(int)]
K = 40  # look ahead K inserts
nxt_dt = np.full(n, np.inf); nxt_j = np.full(n, -1)
for k in range(1, K + 1):
    j = np.arange(n - k); jj = j + k
    want = np.where(s[j] == 0, p[j] - 1, p[j] + 1)
    hit = (s[jj] == s[j]) & (p[jj] == want) & (nxt_j[j] < 0)
    nxt_j[j[hit]] = jj[hit]; nxt_dt[j[hit]] = t[jj[hit]] - t[j[hit]]
edges = np.r_[0, .5, 1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 50, 100] * 1e-3
for nm, sel in [('calm all', ~R), ('vol all', R), ('calm L90-320', ~R & (v >= 90) & (v <= 320)), ('vol M20-70', R & (v >= 20) & (v <= 70)), ('vol S1-10', R & (v <= 10))]:
    h = np.histogram(nxt_dt[sel], edges)[0] / sel.sum() / np.diff(edges)  # density per s
    print(f'{nm:14s} n={sel.sum():7d}', ' '.join(f'{x:6.1f}' for x in h))
print('edges ms', edges * 1e3)
np.save(C / 'ladder_next.npy', np.c_[nxt_j, nxt_dt])
