"""Sign vs exact S: S - mid, S - touch, past S moves, plus past signs. Logistic LL comparisons by regime."""
import numpy as np, pandas as pd
from pathlib import Path
from scipy.optimize import minimize
C = Path(__file__).parent / 'cache'
tk = pd.read_parquet(C / 'takers_s.parquet'); z = np.load(C / 's_series.npz')
st, ss = z['t'], z['s']
S_at = lambda t: ss[np.maximum(np.searchsorted(st, t) - 1, 0)]  # S at last insert before t (post-impact of earlier ones ~)
t = tk.t.to_numpy(); s = tk.sg.to_numpy().astype(float); S = tk.s_hat.to_numpy(); n = len(t)
X = pd.DataFrame({'S_mid': S - tk.mid.to_numpy(), 'S_bid': S - tk.bid_before.to_numpy(), 'ask_S': tk.ask_before.to_numpy() - S})
for h in [0.1, 0.5, 2, 5, 20]: X[f'dS{h}'] = S - S_at(t - h)
for tau in [0.25, 1, 4]:
    e = np.zeros(n); acc = 0.0
    for i in range(1, n): acc = acc * np.exp(-(t[i] - t[i - 1]) / tau) + s[i - 1]; e[i] = acc
    X[f'ew{tau}'] = e
X['spread'] = tk.ask_before - tk.bid_before
ok = ((~tk.tw) & ((tk.shock_dt < -1) | (tk.shock_dt > 8)) & (tk.s_sd < 0.5)).to_numpy() & np.isfinite(X.to_numpy()).all(1) & (t > 25)
def fit(cols, m):
    A = np.c_[np.ones(m.sum()), X.loc[m, cols].to_numpy()]; sd = A.std(0); sd[0] = 1; A = A / sd; y = (s[m] > 0).astype(float)
    nll = lambda b: np.sum(np.logaddexp(0, A @ b) - y * (A @ b))
    res = minimize(nll, np.zeros(A.shape[1]), method='BFGS'); return -res.fun, res.x * np.r_[1, np.ones(len(cols))]  # std coefs
for R in [0, 1]:
    m = ok & (tk.reg.to_numpy() == R); y = (s[m] > 0)
    l0 = m.sum() * (-(np.mean(y) * np.log(np.mean(y)) + (1 - np.mean(y)) * np.log(1 - np.mean(y))))
    print(f'==== regime {R} n={m.sum()}')
    print('  corr:', {c: round(np.corrcoef(s[m], X.loc[m, c])[0, 1], 3) for c in X.columns})
    for cols in [['S_mid'], ['dS0.1', 'dS0.5', 'dS2', 'dS5', 'dS20'], ['ew0.25', 'ew1', 'ew4'], ['S_mid', 'dS0.1', 'dS0.5', 'dS2', 'dS5', 'dS20'],
                 ['S_mid', 'ew0.25', 'ew1', 'ew4'], ['S_mid', 'dS0.1', 'dS0.5', 'dS2', 'dS5', 'dS20', 'ew0.25', 'ew1', 'ew4']]:
        l, b = fit(cols, m); print(f'  dLL {l + l0:8.1f}  std-coef', dict(zip(cols, np.round(b[1:], 3))))
