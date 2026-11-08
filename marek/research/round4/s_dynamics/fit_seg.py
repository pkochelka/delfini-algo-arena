"""MLE of S-dynamics + observation params on time segments.
Usage: fit_seg.py --segs a,b a,b --free sig lamj bj c theta off eps --set d0=0.05 e=0.5
Parameters (positive, optimised in log): sig lamj bj lamr theta d0 off eps tau c e jv0 wout
"""
import sys, time, argparse
import numpy as np
from scipy import optimize
import smooth_seg as S

ap = argparse.ArgumentParser()
ap.add_argument('--segs', nargs='+'); ap.add_argument('--free', nargs='+'); ap.add_argument('--set', nargs='*', action='extend', default=[])
ap.add_argument('--maxiter', type=int, default=1500)
A = ap.parse_args()
sls = [S.seg(*map(float, s.split(','))) for s in A.segs]
base = dict(sig=0.4, lamj=0.01, bj=1.5, lamr=0.003, theta=0.45, d0=0.05, off=1.0, eps=0.03, tau=2.0, c=0.026, e=0.5, jv0=1e-6, jvq=1e-8, jvn=1e-8, jvm=1e-8, wout=1e-4, pperm=1.0, tdec=15.0)
for kv in A.set:
    k, v = kv.split('='); base[k] = float(v)
def kw_of(v):
    return dict(sig2=v['sig'] ** 2, lamj=v['lamj'], bj=v['bj'], lamr=v['lamr'], theta=v['theta'], d0=v['d0'], off=v['off'],
                eps=min(v['eps'], 0.9), tau=v['tau'], c=v['c'], e=v['e'], jv0=v['jv0'], jvq=v['jvq'], jvn=v['jvn'], jvm=v['jvm'], wout=min(v['wout'], 0.9), pperm=min(v['pperm'], 1.0), tdec=v['tdec'])
def unpack(x):
    v = dict(base); v.update(dict(zip(A.free, np.exp(x)))); return v
def f(x):
    return -sum(S.loglik(sl, **kw_of(unpack(x))).sum() for sl in sls)
x0 = np.log([base[k] for k in A.free])
t0 = time.time()
r = optimize.minimize(f, x0, method='Nelder-Mead', options={'maxiter': A.maxiter, 'xatol': 2e-3, 'fatol': 0.1, 'adaptive': True})
v = unpack(r.x)
n = sum(sl.stop - sl.start for sl in sls)
print(A.segs, 'n', n, 'nll %.1f' % r.fun, 'nll/ev %.5f' % (r.fun / n), {k: round(float(v[k]), 4) for k in A.free},
      'fixed', {k: v[k] for k in base if k not in A.free}, 'it', r.nit, 't %.0f' % (time.time() - t0), flush=True)
