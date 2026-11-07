"""Profile-likelihood grid for calm S dynamics: sig x jv0 x (c, e)."""
import numpy as np, itertools, sys
import smooth_seg as S
segs = [(700, 840), (1100, 1280), (2780, 3000)]
sls = [S.seg(a, b) for a, b in segs]
obs = dict(theta=0.455, d0=0.05, eps=0.027, off=1.0, tau=3.0)
def nll(**kw):
    return -sum(S.loglik(sl, **obs, **kw).sum() for sl in sls)
res = []
for e, c in [(0.5, 0.026), (0.5, 0.028), (0.5, 0.030), (0.6, 0.0130), (0.6, 0.0140), (0.6, 0.0150), (0.65, 0.0105)]:
    for sig in [0.0, 0.08, 0.13, 0.2]:
        for jv0 in [0.0, 0.008, 0.012, 0.016, 0.02]:
            v = nll(sig2=sig ** 2, jv0=jv0, c=c, e=e)
            res.append((e, c, sig, jv0, v)); print(e, c, sig, jv0, round(v, 1), flush=True)
res = np.array(res); i = res[:, 4].argmin(); print('BEST', res[i])
np.save(S.prep.common.CACHE / 'grid_calm.npy', res)
