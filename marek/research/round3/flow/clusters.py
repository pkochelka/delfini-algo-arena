"""Fine time structure of maker inserts: gaps between consecutive inserts, by size band and regime."""
import numpy as np, pandas as pd
from pathlib import Path
D = Path(__file__).parent; C = D / 'cache'
r = pd.read_pickle(C / 'rows.pkl'); ins = r[r.op == 0].reset_index(drop=True)
reg = (np.load(C / 'hmm_joint_2.npy')[:, 1] > .5)
t = ins.t.to_numpy(); v = ins.vol.to_numpy(); ioc = ins.ioc.to_numpy(); side = ins.side.to_numpy(); px = ins.px.to_numpy()
g = np.diff(t, prepend=np.nan) * 1e3  # ms gap to previous insert (any)
R = reg[np.clip(t.astype(int), 0, 5999)]
print('gap to previous insert (ms) quantiles, makers only')
for nm, sel in [('calm', ~R), ('vol', R)]:
    m = sel & ~ioc
    print(nm, np.round(np.quantile(g[m][1:], [.001, .01, .05, .1, .5]), 3), 'P(gap<0.05ms)', (g[m] < .05).mean(), 'P(<0.5ms)', (g[m] < .5).mean())
    print('  expected P(gap<0.5ms) under Poisson at rate', round(m.sum() / sel.sum() * 0 + (sel.sum() / (R.mean() * 6000 if nm == 'vol' else (1 - R.mean()) * 6000)), 1))
# histogram of gaps < 5 ms
for nm, sel in [('calm', ~R & ~ioc), ('vol', R & ~ioc)]:
    h, e = np.histogram(g[sel], bins=np.r_[0, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5])
    print(nm, dict(zip(e[:-1], h)))
