"""Sign model comparison (logistic, ex-shock): past signs (EWMA in time) vs past mid moves vs book imbalance, by regime."""
import numpy as np, pandas as pd
from pathlib import Path
from scipy.optimize import minimize
C = Path(__file__).parent / 'cache'
tk = pd.read_parquet(C / 'takers.parquet'); r = pd.read_pickle(C / 'rows.pkl')
rt = r.t.to_numpy(); mid = ((r.bid_after + r.ask_after) / 2).ffill().to_numpy()
def mid_at(t):  # mid just before time t
    i = np.searchsorted(rt, t, 'left') - 1; return mid[np.maximum(i, 0)]
t = tk.t.to_numpy(); s = tk.sg.to_numpy().astype(float); n = len(t)
m0 = mid_at(t)
X = {}
for h in [0.1, 0.5, 2, 5, 15]: X[f'dmid{h}'] = m0 - mid_at(t - h)
for tau in [0.25, 1, 4, 15]:
    e = np.zeros(n); acc = 0.0
    for i in range(1, n):
        acc = acc * np.exp(-(t[i] - t[i - 1]) / tau) + s[i - 1]; e[i] = acc
    X[f'ew{tau}'] = e
bq = tk.bid_before.to_numpy()
X['mid_frac'] = m0 - np.floor(m0)
X = pd.DataFrame(X)
ex = (np.abs(tk.shock_dt) > 25).to_numpy() & np.isfinite(X.to_numpy()).all(1) & (t > 20)
def fit(cols, m):
    A = np.c_[np.ones(m.sum()), X.loc[m, cols].to_numpy()] if cols else np.ones((m.sum(), 1))
    sd = A.std(0); sd[0] = 1; A = A / sd; y = (s[m] > 0).astype(float)
    def nll(b):
        z = A @ b; return np.sum(np.logaddexp(0, z) - y * z)
    res = minimize(nll, np.zeros(A.shape[1]), method='BFGS')
    return -res.fun, res.x / sd
for reg in [0, 1]:
    m = ex & (tk.vol1s.to_numpy() == reg)
    print('==== regime', reg, 'n', m.sum())
    l0, _ = fit([], m)
    for cols in [['ew0.25', 'ew1', 'ew4'], ['dmid0.1', 'dmid0.5', 'dmid2', 'dmid5'], ['ew0.25', 'ew1', 'ew4', 'dmid0.1', 'dmid0.5', 'dmid2', 'dmid5'],
                 ['ew0.25', 'ew1', 'ew4', 'ew15', 'dmid0.1', 'dmid0.5', 'dmid2', 'dmid5', 'dmid15']]:
        l, b = fit(cols, m); print(f'  dLL {l - l0:8.1f}  ', dict(zip(['c'] + cols, np.round(b, 4))))
    print('  corr sign vs', {c: round(np.corrcoef(s[m], X.loc[m, c])[0, 1], 3) for c in X.columns})
