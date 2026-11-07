"""Nonparametric (interval-censored EM) maker offset D distribution per regime, given smoothed S; plus
parametric fit D = d0 + Exp(theta) and D = k*(0.05+Exp(0.45))."""
import numpy as np, pandas as pd, sys
from scipy import optimize
import prep
tag = sys.argv[1] if len(sys.argv) > 1 else 'v2'
d = prep.build(); r = dict(np.load(prep.common.CACHE / f'smooth_{tag}.npz'))
m = r['mean']; k = d['kind']; p = d['price']; sg = d['sign']
vol = r['vol'].astype(bool); sh = r['shock'].astype(bool)
edges = np.arange(0, 8.0001, 0.05); nb = len(edges) - 1
rng = np.random.default_rng(0)
for nm, msk in [('calm', ~vol & ~sh), ('vol', vol & ~sh)]:
    ii = np.where(msk & (k <= 3) & (r['sd'] < 0.2))[0]
    ii = rng.choice(ii, min(len(ii), 150000), replace=False)
    z = np.where(sg[ii] > 0, m[ii] - p[ii], p[ii] - m[ii])   # S-bid or ask-S
    clip = (k[ii] >= 2)
    hi = z; lo = np.where(clip, -1e9, z - 1)
    # overlap matrix
    A = np.clip(np.minimum(hi[:, None], edges[None, 1:]) - np.maximum(lo[:, None], edges[None, :-1]), 0, None) / 0.05
    w = np.full(nb, 1.0 / nb)
    for it in range(300):
        P = A * w; s = P.sum(1, keepdims=True); s[s == 0] = 1
        w = (P / s).mean(0)
    cdf = np.cumsum(w)
    qs = [edges[1:][np.searchsorted(cdf, q)] for q in [0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]]
    mean = (w * (edges[:-1] + 0.025)).sum()
    print(nm, 'n', len(ii), 'D quantiles .1%/1/5/25/50/75/95/99:', np.round(qs, 2), 'mean %.3f' % mean)
    # parametric MLE shifted exponential
    def nll(th):
        d0, sc = th
        if sc <= 0: return 1e18
        F = lambda x: np.where(x > d0, 1 - np.exp(-(x - d0) / sc), 0.0)
        L = F(hi) - F(lo)
        return -np.log(np.maximum(L, 1e-300)).sum()
    rr = optimize.minimize(nll, [0.05 if nm == 'calm' else 0.6, 0.45 if nm == 'calm' else 0.9], method='Nelder-Mead')
    print('   shifted-exp MLE d0 %.4f theta %.4f  (mean %.3f) nll %.1f' % (rr.x[0], rr.x[1], rr.x.sum(), rr.fun))
    # shifted gamma check
    from scipy import stats
    def nllg(th):
        d0, sc, a = th
        if sc <= 0 or a <= 0: return 1e18
        F = lambda x: stats.gamma.cdf(np.maximum(x - d0, 0), a, scale=sc)
        return -np.log(np.maximum(F(hi) - F(lo), 1e-300)).sum()
    rg = optimize.minimize(nllg, [rr.x[0], rr.x[1], 1.0], method='Nelder-Mead')
    print('   shifted-gamma MLE d0 %.4f scale %.4f shape %.3f nll %.1f' % (*rg.x, rg.fun))
    print('   density per 0.25 bin:', np.round(w.reshape(-1, 5).sum(1)[:24], 3))
