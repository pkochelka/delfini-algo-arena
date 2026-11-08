"""Method check on round 2 (truth: permanent impact): does the transient-impact HMM fit report pperm ~ 1 there?
Calm segments of r2 with r2's final obs params; profile NLL over (pperm, tdec) with c re-optimised."""
import os
os.environ['TAPE'] = 'data/round2.csv'; os.environ['LADDER'] = '0'
import numpy as np
from scipy import optimize
import smooth_seg as S
sls = [S.seg(700, 840), S.seg(1100, 1280), S.seg(2780, 3000)]
obs = dict(theta=0.450, d0=0.049, eps=0.022, tau=3.15, off=1.0, sig2=1e-8, lamj=1e-4, jv0=0.0155, e=0.625)
def nll(**kw): return -sum(S.loglik(sl, **obs, **kw).sum() for sl in sls)
for pp, td in [(1.0, 15), (0.8, 15), (0.5, 15), (0.5, 30), (0.5, 60)]:
    r = optimize.minimize_scalar(lambda c: nll(c=c, pperm=pp, tdec=td), bounds=(0.010, 0.022), method='bounded', options={'xatol': 2e-4})
    print('round2 pperm', pp, 'tdec', td, 'c %.4f nll %.1f' % (r.x, r.fun), flush=True)
