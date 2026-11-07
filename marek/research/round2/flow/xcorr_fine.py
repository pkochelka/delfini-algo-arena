"""Fine cross-correlation maker vs taker counts (0.1 s bins, lags -60..60) and 1 s count histograms."""
import numpy as np, pandas as pd
from pathlib import Path
C = Path(__file__).parent / 'cache'
r = pd.read_pickle(C / 'rows.pkl'); ins = r[r.op == 0]
def cnt(t, w, T=6000): return np.bincount((np.asarray(t) / w).astype(int), minlength=int(T / w))[:int(T / w)].astype(float)
w = 0.1
mk = cnt(ins.t[~ins.ioc], w); tk = cnt(ins.t[ins.ioc], w)
a = mk - mk.mean(); b = tk - tk.mean(); n = len(a); den = np.sqrt(a @ a * (b @ b))
def xc(l): return (a[:n - l] @ b[l:] if l >= 0 else a[-l:] @ b[:n + l]) / den
print('corr(mk[t], tk[t+l]) lag in 0.1s')
for l in range(-60, 61, 3): print(f'{l*w:5.1f}s {xc(l):+.4f}', '#' * int(abs(xc(l)) * 800))
m1 = cnt(ins.t[~ins.ioc], 1.0); t1 = cnt(ins.t[ins.ioc], 1.0)
print('maker 1s count quantiles', np.percentile(m1, [0, 1, 5, 25, 50, 75, 95, 99, 100]))
print('taker 1s count quantiles', np.percentile(t1, [0, 1, 5, 25, 50, 75, 95, 99, 100]))
h = np.bincount(m1.astype(int)); 
from scipy.stats import poisson
lam = m1.mean()
for k in range(100, 260, 10): print(k, h[k:k+10].sum(), round(len(m1)*(poisson.cdf(k+9, lam)-poisson.cdf(k-1, lam)), 1))
