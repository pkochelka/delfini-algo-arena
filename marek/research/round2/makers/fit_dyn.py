"""Fit S dynamics + maker offset parameters of the grid HMM by maximum likelihood on time windows."""
import sys, json
import numpy as np
from scipy import optimize
import hmm

def nll(th, t0, t1, fixed):
    p = dict(fixed)
    names = [k for k in ['sigma', 'lam', 'jsd', 'c', 'd0', 'scale', 'taker_off'] if k not in fixed]
    for n, v in zip(names, th):
        p[n] = v
    if min(p['sigma'], p['lam'], p['jsd'], p['scale']) <= 0 or p['d0'] < 0 or p['c'] < 0:
        return 1e12
    r = hmm.run(t0_s=t0, t1_s=t1, smooth=False, **p)
    return -r['ll']

if __name__ == '__main__':
    t0, t1 = float(sys.argv[1]), float(sys.argv[2])
    fixed = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
    init = {'sigma': 0.6, 'lam': 3.0, 'jsd': 1.5, 'c': 0.03, 'd0': 0.05, 'scale': 0.45, 'taker_off': 1.0}
    names = [k for k in init if k not in fixed]
    x0 = [init[k] for k in names]
    r = optimize.minimize(nll, x0, args=(t0, t1, fixed), method='Nelder-Mead',
                          options={'maxfev': 250, 'xatol': 1e-3, 'fatol': 0.5})
    print(json.dumps({'window': [t0, t1], 'fixed': fixed, **dict(zip(names, map(float, r.x))), 'nll': float(r.fun)}))
