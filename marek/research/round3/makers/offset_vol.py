"""Volatile D: exp vs gamma shape (with ladder mixture), S-precision subsets, and away from regime switches."""
import numpy as np, pandas as pd
from scipy import optimize, stats
from offset import g0, prep, gh_w, fit
import states

def nll_g(th, y, ylim, clipped, lad_ok):
    d0, sc, k = th[0], th[1], th[2]; w = 1 / (1 + np.exp(-th[3]))
    if sc <= 0 or d0 < 0 or k <= 0: return 1e18
    F = lambda z: stats.gamma.cdf(np.maximum(z - d0, 0), k, scale=sc)
    pr = np.where(clipped[:, None], F(ylim), F(y) - F(y - 1)); pr = (pr * gh_w).sum(1)
    return -np.log(np.maximum((1 - w) * pr + w * lad_ok / 6, 1e-300)).sum()
v = g0[g0.st == 'vol']
# distance from nearest regime switch
R = states.REG; sw = R.start.to_numpy()[1:] * 1e6
d = np.abs(v.t_us.to_numpy()[:, None] - sw[None, :]).min(1) / 1e6
for name, gg in [('vol all', v.sample(60000, random_state=0)), ('vol S_sd<0.08', v[v.S_sd < 0.08].sample(60000, random_state=0)),
                 ('vol >10 s from switch', v[d > 10].sample(60000, random_state=0))]:
    x, l, cs = fit(gg)
    a = prep(gg)
    r = optimize.minimize(nll_g, [0.1, 1.0, 1.3, -4], args=a, method='Nelder-Mead', options={'maxiter': 1500, 'xatol': 1e-4, 'fatol': 1e-3})
    print(f'{name}: S_sd med {gg.S_sd.median():.3f} exp d0 {x[0]:.3f} sc {x[1]:.3f} mean {x[0]+x[1]:.3f} nll {l:.4f} | gamma d0 {r.x[0]:.3f} sc {r.x[1]:.3f} k {r.x[2]:.3f} mean {r.x[0]+r.x[1]*r.x[2]:.3f} nll {r.fun/len(gg):.4f}')
