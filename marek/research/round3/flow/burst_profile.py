"""Fine post-jump taker profile split into jump-direction vs opposite takers (rate per shock per s), r3 and r2."""
import numpy as np, pandas as pd
from pathlib import Path
D = Path(__file__).parent; C = D / 'cache'
X = pd.read_csv(D / 'shock_events.csv'); r = pd.read_pickle(C / 'rows.pkl'); k = r[(r.op == 0) & r.ioc]
R2 = D.parents[1] / 'round2'; e2 = pd.read_csv(R2 / 'flow/shock_events.csv'); r2 = pd.read_pickle(R2 / 'flow/cache/rows.pkl')
S2 = pd.read_parquet(R2 / 's_dynamics/s_path.parquet'); i2 = r2[r2.op == 0]; t2 = i2.t.to_numpy(); s2 = S2.s_hat.to_numpy()
def S2at(x): return s2[np.clip(np.searchsorted(t2, x, 'right') - 1, 0, None)]
J2 = np.array([S2at(o + 3.4) - S2at(o + 2.8) for o in e2.onset]); k2 = r2[(r2.op == 0) & r2.ioc]
edges = np.r_[np.arange(-0.2, 1.0, 0.05), np.arange(1.0, 4.01, 0.25)]
def prof(tj, J, k):
    tk = k.t.to_numpy(); sg = np.where(k.side.to_numpy() == 0, 1, -1); hd = np.zeros(len(edges) - 1); ho = hd.copy()
    for t0, j in zip(tj, J):
        if abs(j) < 3: continue
        m = (tk > t0 + edges[0]) & (tk < t0 + edges[-1]); d = tk[m] - t0; same = sg[m] == np.sign(j)
        hd += np.histogram(d[same], edges)[0]; ho += np.histogram(d[~same], edges)[0]
    n = (np.abs(J) >= 3).sum(); w = np.diff(edges); return hd / n / w, ho / n / w, n
a = prof(X.t_jump.to_numpy(), X.jump.to_numpy(), k); b = prof(e2.onset.to_numpy() + 3.0, J2, k2)
print(f'rates per shock per s; r3 n={a[2]}, r2 n={b[2]} (|J|>=3)')
print('   bin      r3 dir  r3 opp | r2 dir  r2 opp')
for i in range(len(edges) - 1):
    print(f'{edges[i]:+5.2f}-{edges[i + 1]:+5.2f} {a[0][i]:6.1f} {a[1][i]:6.1f} | {b[0][i]:6.1f} {b[1][i]:6.1f}')
def extra(p, lo, hi, base):
    m = (edges[:-1] >= lo) & (edges[1:] <= hi); return ((p[m] - base) * np.diff(edges)[m]).sum()
for nm, p in [('r3', a), ('r2', b)]:
    print(f'{nm}: extra dir takers 0-0.2 s {extra(p[0], 0, 0.2, 4):.1f}, extra opp 0-0.2 s {extra(p[1], 0, 0.2, 4):.1f}; '
          f'extra dir 0.2-4 s {extra(p[0], 0.2, 4, 4):.1f}, extra opp 0.2-4 s {extra(p[1], 0.2, 4, 4):.1f}')
