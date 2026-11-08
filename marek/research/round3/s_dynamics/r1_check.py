"""Method check on round 1 (truth: permanent impact c=0.026, e=0.5, no noise): transient-impact profile must prefer pperm=1."""
import os
os.environ['TAPE'] = 'data/round1.csv'; os.environ['LADDER'] = '0'
from scipy import optimize
import smooth_seg as S
sls = [S.seg(1000, 1300), S.seg(5000, 5200)]
obs = dict(theta=0.45, d0=0.05, eps=0.001, off=1.0, tau=3.0, sig2=1e-8, lamj=1e-4, jv0=0.002, e=0.5)
def nll(**kw): return -sum(S.loglik(sl, **obs, **kw).sum() for sl in sls)
for pp, td in [(1.0, 15), (0.8, 15), (0.5, 15), (0.5, 30)]:
    r = optimize.minimize_scalar(lambda c: nll(c=c, pperm=pp, tdec=td), bounds=(0.018, 0.034), method='bounded', options={'xatol': 2e-4})
    print('round1 pperm', pp, 'tdec', td, 'c %.4f nll %.1f' % (r.x, r.fun), flush=True)
