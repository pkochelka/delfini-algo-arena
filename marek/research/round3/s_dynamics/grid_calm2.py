import numpy as np, sys
from scipy import optimize
import smooth_seg as S
segs = [(700, 840), (1100, 1280), (2780, 3000)]
sls = [S.seg(a, b) for a, b in segs]
obs = dict(theta=0.455, d0=0.05, eps=0.027, off=1.0, tau=3.0)
def nll(**kw): return -sum(S.loglik(sl, **obs, **kw).sum() for sl in sls)
for e in [0.5, 0.55, 0.6, 0.65, 0.7]:
    c0 = 0.028 * 60 ** (0.5 - e)  # match at f~60
    r = optimize.minimize_scalar(lambda c: nll(sig2=0.1 ** 2, jv0=0.016, c=c, e=e), bounds=(c0 * 0.7, c0 * 1.4), method='bounded', options={'xatol': 1e-4})
    print('e', e, 'c %.5f' % r.x, 'nll %.1f' % r.fun, flush=True)
