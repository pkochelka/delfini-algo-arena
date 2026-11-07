"""Detect 'shock' events: maker-rate drought followed by a taker burst. Profiles aligned on drought onset."""
import numpy as np, pandas as pd
from pathlib import Path
D = Path(__file__).parent; C = D / 'cache'
r = pd.read_pickle(C / 'rows.pkl'); ins = r[r.op == 0]
mk = ins[~ins.ioc]; tk = ins[ins.ioc]
w = 0.05
def cnt(t, T=6000): return np.bincount((np.asarray(t) / w).astype(int), minlength=int(T / w))[:int(T / w)].astype(float)
m = cnt(mk.t)
# rolling 0.5 s maker count
k = int(0.5 / w); roll = np.convolve(m, np.ones(k), 'valid')  # roll[i] = sum m[i:i+k]
low = roll < 50
# group into episodes
idx = np.where(low)[0]
eps = []
for i in idx:
    if not eps or i > eps[-1][1] + 20: eps.append([i, i])
    else: eps[-1][1] = i
print('episodes', len(eps))
# refine onset/offset with a change-point on maker times: find onset as start of window, end as last low window + k
rows = []
mt = mk.t.to_numpy()
for a, b in eps:
    t0, t1 = a * w, (b + k) * w
    # refine: within [t0-1, t1+1] choose segment [s,e] maximizing Poisson LR for rate drop
    sel = mt[(mt > t0 - 1.5) & (mt < t1 + 1.5)]
    best = None
    grid = np.arange(t0 - 1.5, t1 + 1.5, 0.01)
    cs = np.searchsorted(sel, grid)
    lam0 = 178.3
    for i in range(len(grid)):
        for j in range(i + 50, len(grid), 2):
            n = cs[j] - cs[i]; L = grid[j] - grid[i]; lam = n / L
            ll = n * np.log(lam / lam0) - (lam - lam0) * L if n > 0 else lam0 * L
            if best is None or ll > best[0]: best = (ll, grid[i], grid[j], lam)
    rows.append(dict(onset=best[1], end=best[2], dur=best[2] - best[1], maker_rate=best[3]))
ev = pd.DataFrame(rows)
ev['gap'] = ev.onset.diff()
pd.set_option('display.width', 200)
print(ev.round(3).to_string())
print(ev.describe().round(3).to_string())
ev.to_csv(D / 'shock_events.csv', index=False)
