"""Joint (side, tail) model comparison. M0: side ~ Phi(x/sig), tail ~ Bernoulli(q_r) independent.
M1 (noisy valuation): V = S + e, e ~ N(0, sig); side = sign(V - mid); tail iff |V - mid| > theta_r.
M2: side ~ Phi(x/sig), logit P(tail) = a_r + b_r * xa (xa = sg*x), free."""
import numpy as np, pandas as pd
from pathlib import Path
from scipy.optimize import minimize
from scipy.stats import norm
C = Path(__file__).parent / 'cache'
tk = pd.read_parquet(C / 'takers_s.parquet')
S = tk.s_hat.to_numpy(); s = tk.sg.to_numpy(); reg = tk.reg.to_numpy(); x = S - tk.mid.to_numpy()
k = np.where(reg == 1, 2.55, 1.0); tail = (s * (tk.px.to_numpy() - S) - k > 0.5)
ok = ((~tk.tw) & ((tk.shock_dt < -1) | (tk.shock_dt > 8)) & (tk.s_sd < 0.3)).to_numpy() & np.isfinite(x) & (tk.t > 25).to_numpy()
x, s, reg, tail = x[ok], s[ok], reg[ok], tail[ok]; xa = s * x
def M0(b):
    sig = np.exp(b[0]); q = 1 / (1 + np.exp(-b[1:3]))[reg]
    return -(norm.logcdf(xa / sig).sum() + np.where(tail, np.log(q), np.log(1 - q)).sum())
def M1(b):
    sig = np.exp(b[0]); th = np.exp(b[1:3])[reg]
    # P(side, tail) : side aligned with V-mid sign s: V-mid = x + e; s*(x+e) > 0 ; tail iff s*(x+e) > th
    p_side = norm.cdf(xa / sig); p_tail = norm.cdf((xa - th) / sig)
    return -np.where(tail, np.log(np.maximum(p_tail, 1e-300)), np.log(np.maximum(p_side - p_tail, 1e-300))).sum()
def M2(b):
    sig = np.exp(b[0]); z = b[1:3][reg] + b[3:5][reg] * xa; lq = -np.logaddexp(0, -z); l1q = -np.logaddexp(0, z)
    return -(norm.logcdf(xa / sig).sum() + np.where(tail, lq, l1q).sum())
def M3(b):  # M1 with separate sigma for tail threshold noise: tail iff s*(x+e2)>th, e2 indep N(0,sig2)  (just a free-er variant)
    sig = np.exp(b[0]); th = np.exp(b[1:3])[reg]; sig2 = np.exp(b[3])
    p_side = norm.cdf(xa / sig); p_tail_given = norm.cdf((xa - th) / sig2)
    return -np.where(tail, np.log(np.maximum(p_side * p_tail_given, 1e-300)), np.log(np.maximum(p_side * (1 - p_tail_given), 1e-300))).sum()
for nm, f, b0 in [('M0 indep', M0, [1.5, -3, -2]), ('M1 valuation', M1, [1.5, 2, 2]), ('M2 logit tail(xa)', M2, [1.5, -3, -2, 0, 0]), ('M3 side x tail(xa) probit', M3, [1.5, 2, 2, 1.5])]:
    res = minimize(f, b0, method='Nelder-Mead', options=dict(maxiter=4000, xatol=1e-5, fatol=1e-3))
    print(f'{nm:28s} LL {-res.fun:10.1f}  k {len(b0)}  params', np.round(res.x, 3), ' exp:', np.round(np.exp(res.x[:3]), 3))
