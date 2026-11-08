"""3-state KF on the reconstructed path: S = F + Y1 + Y2.
Deterministic kick mu = c*s*f^e: F += p*mu, Y1 += (1-p)*mu, Y1 decays with T.
Noise xi ~ N(0,q) at filled takers: F += px*xi, Y2 += (1-px)*xi; Y2 decays with Tx and has its own diffusion sY*dt (OU).
F diffusion sF*dt. Regime-specific c,e,q,sY; p,T,px,Tx shared or per regime per variant. Usage: kf3.py TAG variant"""
import sys, os, json, numpy as np, pandas as pd
from numba import njit
from scipy import optimize
import prep
tag = sys.argv[1]; variant = sys.argv[2]
TAPE = os.environ.get('TAPE', 'data/round3.csv')
d = prep.build(TAPE); r = dict(np.load(prep.common.CACHE / f'smooth_{tag}.npz'))
t = d['t']; m = r['mean']; sd = r['sd']; vol = r['vol'].astype(bool)
SH = pd.read_csv(os.environ.get('SHOCKS', str(prep.common.HERE / '../flow/shock_events.csv'))).onset.to_numpy()
OBS = os.environ.get('OBS', 'ioc')
io = np.where(d['ioc'])[0] if OBS == 'ioc' else np.arange(0, len(t), int(OBS))
if OBS != 'ioc':
    io = np.union1d(io, np.where(d['ioc'])[0])
tt = t[io]; z = m[io]; R = sd[io] ** 2; vv = vol[io].astype(np.int64)
sg = d['sign'][io]; ff = np.where(d['ioc'][io], d['filled'][io], 0.0)
reset = np.zeros(len(io), np.int64); skip = np.zeros(len(io), np.int64)
for a in SH:
    skip[(tt >= a + 2.5) & (tt < a + 3.8)] = 1
    j = np.searchsorted(tt, a + 3.8); reset[min(j, len(io) - 1)] = 1


@njit(cache=True)
def kf(tt, z, R, vv, sg, ff, reset, skip, th, evt):
    # th: c0 c1 e0 e1 p0 p1 T0 T1 px0 px1 Tx0 Tx1 q0 q1 sY0 sY1 sF0 sF1 r0
    n = len(tt)
    x = np.zeros(3); x[0] = z[0]
    P = np.zeros((3, 3)); P[0, 0] = 1.0
    ll = 0.0
    for j in range(n):
        if j > 0:
            dt = tt[j] - tt[j - 1]; g = vv[j - 1]
            if evt:   # decay per IOC event instead of per second (T = mean lifetime in IOC events)
                a1 = np.exp(-1.0 / th[6 + g]); a2 = np.exp(-1.0 / th[10 + g])
            else:
                a1 = np.exp(-dt / th[6 + g]); a2 = np.exp(-dt / th[10 + g])
            x[1] *= a1; x[2] *= a2
            A = np.array([1.0, a1, a2])
            for u in range(3):
                for w in range(3):
                    P[u, w] *= A[u] * A[w]
            P[0, 0] += th[16 + g] * dt
            P[2, 2] += th[14 + g] * th[10 + g] / 2 * (1 - a2 * a2)   # exact OU variance over dt (sY = innovation rate)
        if reset[j]:
            P[0, 0] += 1e4
        if not skip[j]:
            pred = x[0] + x[1] + x[2]
            h = P.sum(axis=1)               # P @ [1,1,1]
            Sv = h.sum() + R[j] + th[18]
            inn = z[j] - pred
            ll += -0.5 * (np.log(2 * np.pi * Sv) + inn * inn / Sv)
            K = h / Sv
            x += K * inn
            for u in range(3):
                for w in range(3):
                    P[u, w] -= K[u] * h[w]
        if ff[j] > 0:
            g = vv[j]
            mu = th[0 + g] * sg[j] * ff[j] ** th[2 + g]
            p = th[4 + g]; px = th[8 + g]; q = th[12 + g]
            x[0] += p * mu; x[1] += (1 - p) * mu
            B = np.array([px, 0.0, 1 - px])
            for u in range(3):
                for w in range(3):
                    P[u, w] += B[u] * B[w] * q
    return ll


names = ['c0', 'c1', 'e0', 'e1', 'p0', 'p1', 'T0', 'T1', 'px0', 'px1', 'Tx0', 'Tx1', 'q0', 'q1', 'sY0', 'sY1', 'sF0', 'sF1', 'r0']
base = dict(c0=0.017, c1=0.062, e0=0.616, e1=0.628, p0=0.54, p1=0.55, T0=16.7, T1=13.6, px0=0.9, px1=0.9, Tx0=10., Tx1=10.,
            q0=0.0127, q1=0.05, sY0=0.01, sY1=0.05, sF0=1e-6, sF1=1e-6, r0=1e-6)
VAR = {
    'A': dict(free=['c0', 'c1', 'e0', 'e1', 'p0', 'p1', 'T0', 'T1', 'q0', 'q1', 'r0'], fix=dict(px0=1, px1=1, sY0=0, sY1=0)),
    'B': dict(free=['c0', 'c1', 'e0', 'e1', 'p0', 'p1', 'T0', 'T1', 'q0', 'q1', 'r0', 'sY0', 'sY1', 'Tx0', 'Tx1'], fix=dict(px0=1, px1=1)),
    'C': dict(free=['c0', 'c1', 'e0', 'e1', 'p0', 'p1', 'T0', 'T1', 'q0', 'q1', 'r0', 'px0', 'px1', 'Tx0', 'Tx1'], fix=dict(sY0=0, sY1=0)),
    'D': dict(free=['c0', 'c1', 'e0', 'e1', 'p0', 'p1', 'T0', 'T1', 'q0', 'q1', 'r0', 'px0', 'px1', 'Tx0', 'Tx1', 'sY0', 'sY1'], fix={}),
    'At': dict(free=['c0', 'c1', 'e0', 'e1', 'p0', 'T0', 'q0', 'q1', 'r0'], fix=dict(px0=1, px1=1, sY0=0, sY1=0), tie=True),
    'Aevt': dict(free=['c0', 'c1', 'e0', 'e1', 'p0', 'p1', 'T0', 'T1', 'q0', 'q1', 'r0'], fix=dict(px0=1, px1=1, sY0=0, sY1=0, T0=130., T1=160.), evt=True),
    'Aevt_t': dict(free=['c0', 'c1', 'e0', 'e1', 'p0', 'T0', 'q0', 'q1', 'r0'], fix=dict(px0=1, px1=1, sY0=0, sY1=0, T0=140.), evt=True, tie=True),
    'E': dict(free=['c0', 'c1', 'e0', 'e1', 'p0', 'p1', 'T0', 'T1', 'q0', 'q1', 'r0', 'sF0', 'sF1'], fix=dict(px0=1, px1=1, sY0=0, sY1=0)),
}[variant]
fixed = dict(base); fixed.update(VAR['fix']); free = VAR['free']
bounded = {'p0', 'p1', 'px0', 'px1'}
def unpack(x):
    v = dict(fixed)
    for k, xi in zip(free, x):
        v[k] = 1 / (1 + np.exp(-xi)) if k in bounded else np.exp(xi)
    if VAR.get('tie'):
        v['p1'] = v['p0']; v['T1'] = v['T0']
    return v
def nll(x):
    v = unpack(x)
    return -kf(tt, z, R, vv, sg, ff, reset, skip, np.array([v[k] for k in names], float), bool(VAR.get('evt', False)))
x0 = [np.log(fixed[k] / (1 - fixed[k])) if k in bounded else np.log(fixed[k]) for k in free]
res = optimize.minimize(nll, x0, method='Nelder-Mead', options={'maxiter': 30000, 'maxfev': 30000, 'xatol': 1e-4, 'fatol': 1e-3, 'adaptive': True})
res = optimize.minimize(nll, res.x, method='Nelder-Mead', options={'maxiter': 30000, 'maxfev': 30000, 'xatol': 1e-5, 'fatol': 1e-4, 'adaptive': True})
v = unpack(res.x)
print(TAPE, tag, 'OBS', OBS, variant, 'nll %.1f' % res.fun, 'k', len(free), {k: round(float(v[k]), 5) for k in names}, flush=True)
json.dump(dict(variant=variant, nll=res.fun, params={k: float(v[k]) for k in names}), open(prep.common.CACHE / f'kf3_{tag}_{variant}_{OBS}.json', 'w'))
