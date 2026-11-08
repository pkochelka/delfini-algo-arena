"""Kalman-filter MLE of the S dynamics on the reconstructed path (S_hat is precise: posterior sd ~0.07).
State (F, Y): S = F + Y. At filled taker k: kick = c*sign*f^e + xi, xi ~ N(0, q); F += p*c*s*f^e + pxi*xi,
Y += (1-p)*c*s*f^e + (1-pxi)*xi. Between takers Y decays exp(-dt/T); optional F diffusion sF2*dt.
Observations: S_hat just before every IOC (var = s_sd^2 + r0). Shock: F variance reset after each jump.
Usage: kf_fit.py TAG [variant]. Variants fix/free parameters (see bottom)."""
import sys, os, json, numpy as np, pandas as pd
from numba import njit
from scipy import optimize
import prep
tag = sys.argv[1]; variant = sys.argv[2] if len(sys.argv) > 2 else 'full'
TAPE = os.environ.get('TAPE', 'data/round3.csv')
d = prep.build(TAPE); r = dict(np.load(prep.common.CACHE / f'smooth_{tag}.npz'))
t = d['t']; m = r['mean']; sd = r['sd']; vol = r['vol'].astype(bool); sh = r['shock'].astype(bool)
SHF = os.environ.get('SHOCKS', str(prep.common.HERE / '../flow/shock_events.csv'))
SH = pd.read_csv(SHF).onset.to_numpy()
io = np.where(d['ioc'])[0]
tt = t[io]; z = m[io]; R = sd[io] ** 2; vv = vol[io].astype(np.int64)
sg = d['sign'][io]; ff = d['filled'][io]
# reset flag: first observation after each shock jump (onset+3.0 .. take onset+3.6) ; skip obs inside [onset+2.5, onset+3.6]
reset = np.zeros(len(io), np.int64); skip = np.zeros(len(io), np.int64)
for a in SH:
    skip[(tt >= a + 2.5) & (tt < a + 3.8)] = 1
    j = np.searchsorted(tt, a + 3.8); reset[min(j, len(io) - 1)] = 1


@njit(cache=True)
def kf(tt, z, R, vv, sg, ff, reset, skip, c0, c1, e0, e1, p0, p1, px0, px1, T0, T1, q0, q1, sF0, sF1, r0):
    n = len(tt)
    F = z[0]; Y = 0.0
    P00 = 1.0; P01 = 0.0; P11 = 0.1
    ll = 0.0; nobs = 0
    for j in range(n):
        if j > 0:
            dt = tt[j] - tt[j - 1]
            T = T1 if vv[j - 1] else T0
            a = np.exp(-dt / T)
            Y = Y * a; P11 = P11 * a * a; P01 = P01 * a
            P00 += (sF1 if vv[j - 1] else sF0) * dt
        if reset[j]:
            P00 += 1e4
        if not skip[j]:
            # observe S = F + Y
            pred = F + Y
            Sv = P00 + 2 * P01 + P11 + R[j] + r0
            innov = z[j] - pred
            ll += -0.5 * (np.log(2 * np.pi * Sv) + innov * innov / Sv)
            nobs += 1
            K0 = (P00 + P01) / Sv; K1 = (P01 + P11) / Sv
            F += K0 * innov; Y += K1 * innov
            a00 = P00 - K0 * (P00 + P01); a01 = P01 - K0 * (P01 + P11); a11 = P11 - K1 * (P01 + P11)
            P00 = a00; P01 = a01; P11 = a11
        if ff[j] > 0:
            if vv[j]:
                mu = c1 * sg[j] * ff[j] ** e1; p = p1; px = px1; q = q1
            else:
                mu = c0 * sg[j] * ff[j] ** e0; p = p0; px = px0; q = q0
            F += p * mu; Y += (1 - p) * mu
            P00 += px * px * q; P01 += px * (1 - px) * q; P11 += (1 - px) * (1 - px) * q
    return ll, nobs


names = ['c0', 'c1', 'e0', 'e1', 'p0', 'p1', 'px0', 'px1', 'T0', 'T1', 'q0', 'q1', 'sF0', 'sF1', 'r0']
base = dict(c0=0.017, c1=0.07, e0=0.61, e1=0.61, p0=0.5, p1=0.5, px0=0.5, px1=0.5, T0=15., T1=15., q0=0.015, q1=0.08,
            sF0=0.0, sF1=0.0, r0=0.001)
V = {
    'full': ['c0', 'c1', 'e0', 'e1', 'p0', 'p1', 'px0', 'px1', 'T0', 'T1', 'q0', 'q1', 'r0'],
    'perm': ['c0', 'c1', 'e0', 'e1', 'q0', 'q1', 'r0', 'sF0', 'sF1'],          # r2 model: p = px = 1
    'detonly': ['c0', 'c1', 'e0', 'e1', 'p0', 'p1', 'T0', 'T1', 'q0', 'q1', 'r0'],  # xi permanent (px=1)
    'tied': ['c0', 'c1', 'e0', 'e1', 'p0', 'T0', 'q0', 'q1', 'r0'],            # p = px, same p & T in both regimes
    'tied_sF': ['c0', 'c1', 'e0', 'e1', 'p0', 'T0', 'q0', 'q1', 'r0', 'sF0', 'sF1'],
    'full_sF': ['c0', 'c1', 'e0', 'e1', 'p0', 'p1', 'px0', 'px1', 'T0', 'T1', 'q0', 'q1', 'r0', 'sF0', 'sF1'],
}[variant]
fixed = dict(base)
if variant == 'perm':
    fixed.update(p0=1.0, p1=1.0, px0=1.0, px1=1.0)
if variant == 'detonly':
    fixed.update(px0=1.0, px1=1.0)
bounded = {'p0', 'p1', 'px0', 'px1'}


def unpack(x):
    v = dict(fixed)
    for k, xi in zip(V, x):
        v[k] = 1 / (1 + np.exp(-xi)) if k in bounded else np.exp(xi)
    if variant.startswith('tied'):
        v['p1'] = v['p0']; v['px0'] = v['p0']; v['px1'] = v['p0']; v['T1'] = v['T0']
    return v


def nll(x):
    v = unpack(x)
    ll, n = kf(tt, z, R, vv, sg, ff, reset, skip, *[v[k] for k in names])
    return -ll


x0 = [np.log(base[k] / (1 - base[k])) if k in bounded else np.log(max(base[k], 1e-5)) for k in V]
if 'sF0' in V:
    x0[V.index('sF0')] = np.log(1e-3); x0[V.index('sF1')] = np.log(1e-3)
res = optimize.minimize(nll, x0, method='Nelder-Mead', options={'maxiter': 20000, 'maxfev': 20000, 'xatol': 1e-4, 'fatol': 1e-3, 'adaptive': True})
res = optimize.minimize(nll, res.x, method='Nelder-Mead', options={'maxiter': 20000, 'maxfev': 20000, 'xatol': 1e-5, 'fatol': 1e-4, 'adaptive': True})
v = unpack(res.x)
print(TAPE, tag, variant, 'nll %.1f' % res.fun, 'k', len(V), {k: round(float(v[k]), 5) for k in names}, flush=True)
json.dump(dict(variant=variant, nll=res.fun, params={k: float(v[k]) for k in names}),
          open(prep.common.CACHE / f'kf_{tag}_{variant}.json', 'w'))
