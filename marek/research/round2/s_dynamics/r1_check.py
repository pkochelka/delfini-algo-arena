"""Method check on round 1 (known truth: c=0.026, e=0.5, no noise): does the HMM recover e=0.5 and jv0~0?"""
import os
os.environ['TAPE'] = 'data/round1.csv'
import numpy as np
from scipy import optimize
import smooth_seg as S
sls = [S.seg(1000, 1300), S.seg(5000, 5200)]
obs = dict(theta=0.45, d0=0.05, eps=0.001, off=1.0, tau=3.0)
def nll(**kw): return -sum(S.loglik(sl, **obs, **kw).sum() for sl in sls)
for e in [0.5, 0.6, 0.65]:
    c0 = 0.026 * 60 ** (0.5 - e)
    r = optimize.minimize_scalar(lambda c: nll(sig2=0.05 ** 2, jv0=0.002, c=c, e=e), bounds=(c0 * 0.7, c0 * 1.4), method='bounded', options={'xatol': 1e-4})
    print('round1 e', e, 'c %.5f nll %.1f' % (r.x, r.fun), flush=True)
for jv0 in [0.0, 0.002, 0.008, 0.016]:
    for sig in [0.0, 0.1]:
        print('round1 e=.5 c=.026 jv0', jv0, 'sig', sig, 'nll %.1f' % nll(sig2=sig ** 2, jv0=jv0, c=0.026, e=0.5), flush=True)
