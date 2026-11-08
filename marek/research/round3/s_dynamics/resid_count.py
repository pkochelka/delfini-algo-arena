"""Short-window residual variance vs counts (filled takers, unfilled takers, sum filled^(2e)): where does the noise enter?"""
import sys, json, numpy as np, pandas as pd
import prep, smooth_seg as SS
tag, pf = sys.argv[1], sys.argv[2]; W = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
P = json.load(open(prep.common.HERE / pf))
d = SS.d; r = dict(np.load(prep.common.CACHE / f'smooth_{tag}.npz'))
t = d['t']; m = r['mean']; vol = r['vol'].astype(bool); sh = r['shock'].astype(bool)
def arr(k, dflt): return np.where(vol, P['vol'].get(k, dflt), P['calm'].get(k, dflt)).astype(float)
shift = SS.impact(slice(0, len(t)), arr('c', 0), arr('e', 0.6), arr('pperm', 1.0), arr('tdec', 15.0))
E = m - np.cumsum(np.r_[0, shift[:-1]])
nb = int(6000 / W); i = np.clip(np.searchsorted(t, np.arange(nb + 1) * W, 'right') - 1, 0, len(t) - 1)
y = np.diff(E[i]) ** 2
bi = np.minimum((t / W).astype(int), nb - 1)
tk = d['ioc']; f = d['filled']
nf = np.bincount(bi, tk & (f > 0), nb); nu = np.bincount(bi, tk & (f == 0), nb)
f2 = np.bincount(bi, np.where(tk & (f > 0), f ** 1.22, 0), nb)
vb = np.bincount(bi, vol, nb) / np.maximum(np.bincount(bi, minlength=nb), 1)
shb = np.bincount(bi, sh, nb) > 0
for nm, ok in (('calm', (vb == 0) & ~shb), ('vol', (vb == 1) & ~shb)):
    X = np.c_[np.ones(ok.sum()), nf[ok], nu[ok]]
    b, *_ = np.linalg.lstsq(X, y[ok], rcond=None)
    X2 = np.c_[np.ones(ok.sum()), nf[ok], f2[ok]]
    b2, *_ = np.linalg.lstsq(X2, y[ok], rcond=None)
    print(nm, 'W', W, 'E[dE^2] = %.4f + %.4f*n_filled + %.4f*n_unfilled' % tuple(b), '| alt: %.4f + %.4f*n_f + %.6f*sum f^1.22' % tuple(b2))
    for k in range(0, 6):
        s = ok & (nf == k)
        if s.sum() > 30: print('   n_filled=%d  mean dE^2 %.4f  (n %d)' % (k, y[s].mean(), s.sum()))
