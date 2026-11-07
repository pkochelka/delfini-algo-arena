import numpy as np
import smooth_seg as S
sls = [S.seg(a, b) for a, b in [(700, 840), (1100, 1280), (2780, 3000)]]
obs = dict(theta=0.455, d0=0.05, eps=0.027, off=1.0, tau=3.0, c=0.0143, e=0.625)
def nll(**kw): return -sum(S.loglik(sl, **obs, **kw).sum() for sl in sls)
for jv0, jvm, jvq, sig in [(0.016, 0, 0, 0), (0.016, 0, 0, 0.1), (0.0, 0.0008, 0, 0), (0.008, 0.0004, 0, 0), (0.016, 0.0001, 0, 0), (0.012, 0, 0.00003, 0), (0.0, 0, 0.00015, 0)]:
    print('jv0', jv0, 'jvm', jvm, 'jvq', jvq, 'sig', sig, 'nll %.1f' % nll(jv0=jv0, jvm=jvm, jvq=jvq, sig2=sig ** 2), flush=True)
