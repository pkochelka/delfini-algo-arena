"""Taker sign persistence: ACF by lag (all / calm / vol / ex-shock), sign-pair correlation vs time separation."""
import numpy as np, pandas as pd
from pathlib import Path
C = Path(__file__).parent / 'cache'
tk = pd.read_parquet(C / 'takers.parquet')
ex = np.abs(tk.shock_dt) > 25
def acf(s, L=20):
    s = s - s.mean(); return np.array([np.mean(s[:-k] * s[k:]) / s.var() for k in range(1, L + 1)])
for name, m in [('all', np.ones(len(tk), bool)), ('exshock', ex), ('calm', ex & (tk.vol1s == 0)), ('vol', ex & (tk.vol1s == 1))]:
    s = tk.sg.to_numpy()[m].astype(float)
    print(name, len(s), 'mean', round(s.mean(), 4), 'acf', np.round(acf(s, 10), 3), 'lag20,50', np.round(acf(s, 50)[[19, 49]], 3))
# pair correlation vs time separation (ex-shock), all pairs within 60 s
t = tk.t.to_numpy()[ex]; s = tk.sg.to_numpy()[ex].astype(float); s -= s.mean()
edges = np.array([0, .01, .05, .1, .25, .5, 1, 2, 3, 5, 7.5, 10, 15, 20, 30, 45, 60, 90, 120])
num = np.zeros(len(edges) - 1); den = np.zeros(len(edges) - 1)
j0 = 0
for i in range(len(t)):
    j = np.searchsorted(t, t[i] + edges[-1])
    d = t[i + 1:j] - t[i]; b = np.searchsorted(edges, d, 'right') - 1
    num += np.bincount(b, s[i] * s[i + 1:j], len(edges) - 1)[:len(edges) - 1]; den += np.bincount(b, None, len(edges) - 1)[:len(edges) - 1]
print('pair corr vs dt:')
for a, b, c, n in zip(edges[:-1], edges[1:], num / den / s.var(), den): print(f'  {a:6.2f}-{b:6.2f}  {c:+.4f}  n={int(n)}')
