"""Fit the side law P(buy) = F(.) on non-TWAP ex-shock takers. Compare references (mid, microprice, last trade, bid/ask
separately), forms (logistic, probit, mixture informed+noise), lagged S, and regime-specific vs common slope."""
import numpy as np, pandas as pd
from pathlib import Path
from scipy.optimize import minimize
from scipy.stats import norm
C = Path(__file__).parent / 'cache'
tk = pd.read_parquet(C / 'takers_s.parquet'); z = np.load(C / 's_series.npz'); st, ss = z['t'], z['s']
r = pd.read_pickle(C / 'rows.pkl')
S_at = lambda t: ss[np.maximum(np.searchsorted(st, t) - 1, 0)]
t = tk.t.to_numpy(); s = tk.sg.to_numpy(); y = (s > 0).astype(float); S = tk.s_hat.to_numpy()
bid = tk.bid_before.to_numpy(); ask = tk.ask_before.to_numpy(); mid = (bid + ask) / 2
rr = r.loc[r.ioc & (r.op == 0)]
bq = rr.bid_qty_before.to_numpy().astype(float); aq = rr.ask_qty_before.to_numpy().astype(float)
micro = (bid * aq + ask * bq) / np.maximum(aq + bq, 1)
tr = pd.read_pickle(C / 'trades.pkl'); trt = tr.t_us.to_numpy() / 1e6; trp = tr.price.to_numpy()
j = np.searchsorted(trt, t) - 1; last = np.where(j >= 0, trp[np.maximum(j, 0)], np.nan)
reg = tk.reg.to_numpy()
ok = ((~tk.tw) & ((tk.shock_dt < -1) | (tk.shock_dt > 8)) & (tk.s_sd < 0.5)).to_numpy() & np.isfinite(mid + last) & (t > 25)
def ll_logit(Xc, m):
    A = np.c_[np.ones(m.sum()), Xc[m]]; yy = y[m]
    f = lambda b: np.sum(np.logaddexp(0, A @ b) - yy * (A @ b))
    res = minimize(f, np.zeros(A.shape[1]), method='BFGS'); return -res.fun, res.x
def ll_probit(x, m):
    f = lambda b: -np.sum(norm.logcdf((2 * y[m] - 1) * (b[0] + b[1] * x[m])))
    res = minimize(f, [0, .3], method='Nelder-Mead'); return -res.fun, res.x
def ll_mix(x, m):  # pi informed: buy iff x + e > 0 (e ~ N(0, sig)); else coin
    def f(b):
        pi = 1 / (1 + np.exp(-b[0])); sig = np.exp(b[1]); p = (1 - pi) / 2 + pi * norm.cdf(x[m] / sig)
        return -np.sum(y[m] * np.log(p) + (1 - y[m]) * np.log(1 - p))
    res = minimize(f, [0, 0], method='Nelder-Mead'); return -res.fun, [1 / (1 + np.exp(-res.x[0])), np.exp(res.x[1])]
for R in [0, 1, 'all']:
    m = ok & ((reg == R) if R != 'all' else True)
    p0 = y[m].mean(); l0 = m.sum() * (p0 * np.log(p0) + (1 - p0) * np.log(1 - p0))
    print(f'==== regime {R} n={m.sum()}')
    for nm, x in [('S-mid', S - mid), ('S-micro', S - micro), ('S-last', S - last), ('S-mid lagS0.05', S_at(t - 0.05) - mid),
                  ('S-mid lagS0.2', S_at(t - 0.2) - mid), ('(S-mid)/spread', (S - mid) / (ask - bid))]:
        l, b = ll_logit(x[:, None], m); print(f'  logit {nm:16s} dLL {l - l0:7.1f}  slope {b[1]:.3f}')
    l, b = ll_logit(np.c_[S - bid, ask - S], m); print(f'  logit S-bid, ask-S       dLL {l - l0:7.1f}  coefs {np.round(b[1:], 3)}')
    l, b = ll_logit(np.c_[S - mid, S - micro, S - last], m); print(f'  logit mid+micro+last    dLL {l - l0:7.1f}  coefs {np.round(b[1:], 3)}')
    l, b = ll_logit(np.c_[S - mid, np.clip(S - mid, -3, 3), np.tanh(S - mid)], m); print(f'  logit x,clip3,tanh      dLL {l - l0:7.1f}  coefs {np.round(b[1:], 3)}')
    l, b = ll_probit(S - mid, m); print(f'  probit S-mid            dLL {l - l0:7.1f}  b {np.round(b, 3)}')
    l, b = ll_mix(S - mid, m); print(f'  mix pi*Phi(x/sig)+coin  dLL {l - l0:7.1f}  pi {b[0]:.3f} sig {b[1]:.3f}')
