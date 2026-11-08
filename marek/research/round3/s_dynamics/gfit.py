"""Fit G(L) = c*[p + (1-p) exp(-L/T)] (and power-law alternative) to saved propagator curves prop_TAG_B.npz."""
import sys, numpy as np
from scipy import optimize
import prep
tag, B = sys.argv[1], float(sys.argv[2])
z = np.load(prep.common.CACHE / f'prop_{tag}_{B}.npz')
for nm in ('c', 'v'):
    G = z['G' + nm]; se = z['s' + nm]; L = np.arange(len(G)) * B
    sel = L >= 1.0
    f = lambda th: (th[0] * (th[1] + (1 - th[1]) * np.exp(-L / th[2])) - G)[sel] / se[sel]
    r = optimize.least_squares(f, [G[2], 0.5, 15])
    f2 = lambda th: (th[0] * (th[1] + (1 - th[1]) * (1 + L / th[2]) ** -th[3]) - G)[sel] / se[sel]
    r2 = optimize.least_squares(f2, [G[2], 0.3, 5, 1], bounds=([0, 0, 0.01, 0.01], [1, 1, 1e3, 10]))
    print(nm, 'exp: c %.4f p %.3f T %.1f  chi2 %.1f | powerlaw: c %.4f p %.3f L0 %.1f beta %.2f chi2 %.1f (n %d)' % (
        *r.x, (r.fun ** 2).sum(), *r2.x, (r2.fun ** 2).sum(), sel.sum()))
