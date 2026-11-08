"""Do takers chase exogenous S moves? Shock jumps are big S moves not caused by takers. Mean sign*sign(J) of non-TWAP
takers by time after the drought end; also r3 for comparison."""
import numpy as np, pandas as pd, sys
from pathlib import Path
D = Path(__file__).parent; C = D / 'cache'
tk = pd.read_parquet(C / 'takers.parquet'); r = pd.read_pickle(C / 'rows.pkl'); ev = pd.read_csv(D / 'shock_events.csv')
rt = r.t.to_numpy(); mid = ((r.bid_after + r.ask_after) / 2).ffill().to_numpy()
mid_at = lambda t: mid[np.maximum(np.searchsorted(rt, t) - 1, 0)]
J = mid_at(ev.end + 1.0) - mid_at(ev.onset - 0.2)
print('J proxy', np.round(J, 1).tolist())
edges = [-3, -1, 0, .1, .2, .5, 1, 2, 3, 5, 10, 20, 40]
t = tk.t.to_numpy(); s = tk.sg.to_numpy(); nt = ~tk.tw.to_numpy()
rows = []
for k, (e, j) in enumerate(zip(ev.end, J)):
    if abs(j) < 5: continue
    d = t - e
    for a, b in zip(edges[:-1], edges[1:]):
        m = (d >= a) & (d < b) & nt
        rows.append(dict(k=k, a=a, n=m.sum(), al=(s[m] * np.sign(j)).sum()))
R = pd.DataFrame(rows).groupby('a')[['n', 'al']].sum()
R['share_jump_side'] = (1 + R.al / R.n) / 2; R['n_per_shock'] = R.n / (np.abs(J) >= 5).sum()
print(R.round(3))
