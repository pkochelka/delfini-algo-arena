"""MLE per regime of maker sizes outside 2 ms chains: base max(1,round(LN(mu,sig))) + ladder singletons
round(LN(muL, sL)) (sL absorbs the U(.75,1.25) rung noise). Also the base-only fit for chains excluded (all regimes),
and the same fit on round 2 for reference."""
import numpy as np, pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize
o = pd.read_pickle('cache/chains.pkl')
g = np.load('cache/regime_sz.npz')['post']
pv = g[(o.t_us//1_000_000).to_numpy(), 1]
mk = ~o.ioc.astype(bool).to_numpy()
VMAX = 200000
def pmf_rln(mu, s):
    k = np.arange(1, VMAX + 1)
    up = norm.cdf((np.log(k + 0.5) - mu)/s); lo = np.where(k == 1, 0, norm.cdf((np.log(np.maximum(k - 0.5, 1e-9)) - mu)/s))
    return np.maximum(up - lo, 1e-300)
def fit(v, mix=True):
    cnt = np.bincount(np.minimum(v, VMAX), minlength=VMAX + 1)[1:]
    nz = cnt > 0
    def nll(th):
        mu, ls, w, muL, lsL = th if mix else (*th, -30, 0, 0)
        p = pmf_rln(mu, np.exp(ls))
        if mix:
            wl = 1/(1+np.exp(-w)); p = (1-wl)*p + wl*pmf_rln(muL, np.exp(lsL))
        return -(cnt[nz]*np.log(p[nz])).sum()
    x0 = [2.66, np.log(1.8), -3, 4.9, np.log(0.35)] if mix else [2.66, np.log(1.8)]
    r = minimize(nll, x0, method='Nelder-Mead', options=dict(maxiter=4000, xatol=1e-5, fatol=1e-3))
    th = r.x
    out = dict(n=len(v), mu=th[0], sig=np.exp(th[1]), nll=r.fun)
    if mix: out.update(w=1/(1+np.exp(-th[2])), muL=th[3], sL=np.exp(th[4]))
    return out
for name, sel in [('calm', (pv < .05) & mk & (o.L == 1).to_numpy()), ('vol', (pv > .95) & mk & (o.L == 1).to_numpy())]:
    v = o.volume.to_numpy()[sel]
    r0 = fit(v, mix=False); r1 = fit(v, mix=True)
    print(name, 'base-only', {k: round(x, 4) for k, x in r0.items()})
    print(name, 'mixture  ', {k: round(x, 4) for k, x in r1.items()})
    print('   ladder singletons /s ≈', r1['w']*len(v)/((pv < .05).sum() if name == 'calm' else (pv > .95).sum()))
o2 = pd.read_pickle('../../round2/makers/cache/orders.pkl'); v2 = o2[~o2.ioc.astype(bool)].volume.to_numpy()
print('round2 base-only', {k: round(x, 4) for k, x in fit(v2, mix=False).items()})
