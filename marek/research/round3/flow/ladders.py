"""Reconstruct ladder batches: runs of consecutive maker inserts (same side, |dp|=1 step in a consistent direction,
gap < G ms). Compare run-length distribution with the same statistic on a shuffled (Poisson) null."""
import numpy as np, pandas as pd
from pathlib import Path
D = Path(__file__).parent; C = D / 'cache'
r = pd.read_pickle(C / 'rows.pkl'); ins = r[r.op == 0].reset_index(drop=True)
reg = (np.load(C / 'hmm_joint_2.npy')[:, 1] > .5)
mk = ins[~ins.ioc].reset_index(drop=True)
t = mk.t.to_numpy(); v = mk.vol.to_numpy(); s = mk.side.to_numpy(); p = mk.px.to_numpy(); n = len(t)
G = 5e-3
# link i -> i+1 when same side, gap<G, price step of +-1 (away from book: bids step down, asks step up)
step = np.diff(p) * np.where(s[:-1] == 0, -1, 1)  # positive = moving away from touch
link = (s[1:] == s[:-1]) & (np.diff(t) < G) & (np.abs(np.diff(p)) == 1)
# run ids
start = np.r_[True, ~link]; rid = np.cumsum(start) - 1
L = np.bincount(rid)
print('run length dist (all):', np.bincount(L)[:12])
# direction consistency in runs >=3
R = reg[t.astype(int)]
dfr = pd.DataFrame({'rid': rid, 't': t, 'v': v, 's': s, 'p': p, 'R': R})
g = dfr.groupby('rid')
agg = g.agg(n=('t', 'size'), t0=('t', 'first'), t1=('t', 'last'), vmin=('v', 'min'), vmax=('v', 'max'), vmed=('v', 'median'), R=('R', 'first'), p0=('p', 'first'), p1=('p', 'last'), s=('s', 'first'))
big = agg[agg.n >= 3]
print('runs>=3: n', len(big), 'by regime', big.groupby('R').size().to_dict())
print('length dist by regime:'); print(big.groupby('R').n.value_counts().unstack(0).head(12))
big = big.assign(dur_ms=(big.t1 - big.t0) * 1e3, dirn=np.sign((big.p1 - big.p0) * np.where(big.s == 0, -1, 1)))
print('direction (+ away from touch):', big.dirn.value_counts().to_dict())
print('duration ms quantiles', np.round(np.quantile(big.dur_ms, [.1, .5, .9, .99]), 2))
print('vmed quantiles by regime'); print(big.groupby('R').vmed.describe())
agg.to_pickle(C / 'runs.pkl')
