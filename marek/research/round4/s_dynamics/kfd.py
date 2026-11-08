"""Kalman filter on a smoothed S path, observed at IOC rows: S = F + Y + (drift integral in F).
Kick mu = c*sign*f^e at filled takers: F += p*mu, Y += (1-p)*mu; Y decays with T. xi ~ N(0,q) per filled taker
(goes to F with weight px and to Y with 1-px). Drift state D (OU, time const Tm, stationary sd sm): dF = D dt.
Diffusion sF on F. Per-regime params (0 calm, 1 vol). Usage: kfd.py TAG variant  -> _cache/kfd_TAG_variant.json"""
import sys, os, json, numpy as np, pandas as pd
from numba import njit
from scipy import optimize
import prep
tag = sys.argv[1]; variant = sys.argv[2]
d = prep.build(); r = dict(np.load(prep.common.CACHE / f'smooth_{tag}.npz'))
t = d['t']; m = r['mean']; sd = r['sd']; vol = r['vol'].astype(bool)
SH = pd.read_csv(prep.common.HERE / '../flow/shock_events.csv').onset.to_numpy()
io = np.where(d['ioc'])[0]
tt = t[io]; z = m[io]; R = sd[io] ** 2; vv = vol[io].astype(np.int64)
sg = d['sign'][io]; ff = d['filled'][io].astype(float)
reset = np.zeros(len(io), np.int64); skip = np.zeros(len(io), np.int64)
for a in SH:
    skip[(tt >= a + 2.5) & (tt < a + 3.8)] = 1
    j = np.searchsorted(tt, a + 3.8); reset[min(j, len(io) - 1)] = 1
names = ['c0', 'c1', 'e0', 'e1', 'p0', 'p1', 'T0', 'T1', 'px0', 'px1', 'q0', 'q1', 'sF0', 'sF1', 'Tm0', 'Tm1', 'sm0', 'sm1', 'r0', 'k0', 'k1']


@njit(cache=True)
def kf(tt, z, R, vv, sg, ff, reset, skip, th, out, XP, PP, XF, PF, AA):
    n = len(tt)
    x = np.zeros(3); x[0] = z[0]
    P = np.zeros((3, 3)); P[0, 0] = 1.0; P[2, 2] = th[16] ** 2
    A = np.zeros((3, 3)); Q = np.zeros((3, 3)); Pn = np.zeros((3, 3))
    ll = 0.0
    for j in range(n):
        if j > 0:
            dt = tt[j] - tt[j - 1]; g = vv[j - 1]
            aY = np.exp(-dt / th[6 + g]); am = np.exp(-dt / th[14 + g])
            # state transition: F += D*dt ; Y *= aY ; D *= am
            for u in range(3):
                for w in range(3):
                    A[u, w] = 0.0
            A[0, 0] = 1.0; A[0, 2] = th[14 + g] * (1 - am); A[1, 1] = aY; A[2, 2] = am
            x = A @ x
            for u in range(3):
                for w in range(3):
                    Q[u, w] = 0.0
            sm2 = th[16 + g] ** 2
            Q[2, 2] = sm2 * (1 - am * am)
            Q[0, 0] = th[12 + g] ** 2 * dt + sm2 * (1 - am * am) * dt * dt / 3.0
            Q[0, 2] = sm2 * (1 - am * am) * dt / 2.0; Q[2, 0] = Q[0, 2]
            Pn = A @ P @ A.T + Q
            P = Pn
            AA[j] = A
        if reset[j]:
            P[0, 0] += 1e4
        XP[j] = x; PP[j] = P
        if not skip[j]:
            pred = x[0] + x[1]
            h0 = P[0, 0] + P[0, 1]; h1 = P[1, 0] + P[1, 1]; h2 = P[2, 0] + P[2, 1]
            Sv = h0 + h1 + R[j] + th[18]
            inn = z[j] - pred
            ll += -0.5 * (np.log(2 * np.pi * Sv) + inn * inn / Sv)
            K0 = h0 / Sv; K1 = h1 / Sv; K2 = h2 / Sv
            x[0] += K0 * inn; x[1] += K1 * inn; x[2] += K2 * inn
            hh = np.array([h0, h1, h2]); KK = np.array([K0, K1, K2])
            for u in range(3):
                for w in range(3):
                    P[u, w] -= KK[u] * hh[w]
        out[j, 0] = x[2]; out[j, 1] = x[1]
        XF[j] = x; PF[j] = P
        if ff[j] > 0:
            g = vv[j]
            mu = th[0 + g] * sg[j] * ff[j] ** th[2 + g]
            p = th[4 + g]; px = th[8 + g]; q = th[10 + g]
            x[0] += p * mu; x[1] += (1 - p) * mu; x[2] += th[19 + g] * mu
            P[0, 0] += px * px * q; P[1, 1] += (1 - px) ** 2 * q; P[0, 1] += px * (1 - px) * q; P[1, 0] += px * (1 - px) * q
    return ll


base = dict(c0=0.017, c1=0.062, e0=0.616, e1=0.628, p0=0.5, p1=0.55, T0=16.7, T1=13.6, px0=1., px1=1.,
            q0=0.017, q1=0.07, sF0=1e-4, sF1=1e-4, Tm0=5., Tm1=5., sm0=1e-6, sm1=1e-6, r0=1e-6, k0=0.0, k1=0.0)
VAR = {
    'A': dict(free=['c0', 'c1', 'e0', 'e1', 'p0', 'p1', 'T0', 'T1', 'q0', 'q1', 'r0']),
    'Ap1': dict(free=['c0', 'c1', 'e0', 'e1', 'q0', 'q1', 'r0'], fix=dict(p0=1., p1=1.)),
    'AF': dict(free=['c0', 'c1', 'e0', 'e1', 'p0', 'p1', 'T0', 'T1', 'q0', 'q1', 'r0', 'sF0', 'sF1']),
    'D': dict(free=['c0', 'c1', 'e0', 'e1', 'p0', 'p1', 'T0', 'T1', 'q0', 'q1', 'r0', 'Tm0', 'Tm1', 'sm0', 'sm1'],
              init=dict(sm0=0.05, sm1=0.5)),
    'DK': dict(free=['c0', 'c1', 'e0', 'e1', 'p0', 'p1', 'T0', 'T1', 'q0', 'q1', 'r0', 'Tm0', 'Tm1', 'sm0', 'sm1', 'k0', 'k1'],
               init=dict(k0=0.05, k1=0.05)),
    'DF': dict(free=['c0', 'c1', 'e0', 'e1', 'p0', 'p1', 'T0', 'T1', 'q0', 'q1', 'r0', 'Tm0', 'Tm1', 'sm0', 'sm1', 'sF0', 'sF1'],
               init=dict(sm0=0.05, sm1=0.5, sF0=0.05, sF1=0.3)),
    'DX': dict(free=['c0', 'c1', 'e0', 'e1', 'p0', 'p1', 'T0', 'T1', 'q0', 'q1', 'r0', 'Tm0', 'Tm1', 'sm0', 'sm1', 'px0', 'px1'],
               init=dict(sm0=0.05, sm1=0.5, px0=0.6, px1=0.6)),
}[variant]
fixed = dict(base); fixed.update(VAR.get('init', {})); fixed.update(VAR.get('fix', {})); free = VAR['free']
if os.environ.get('INIT'):
    fixed.update({k: min(max(v, 1e-8), 1 - 1e-6) if k in ('p0', 'p1', 'px0', 'px1') else (v if k in ('k0', 'k1') else max(v, 1e-8)) for k, v in json.load(open(os.environ['INIT']))['params'].items() if k in free})
bounded = {'p0', 'p1', 'px0', 'px1'}
RAW = {'k0', 'k1'}
def unpack(x):
    v = dict(fixed)
    for k, xi in zip(free, x):
        v[k] = xi if k in RAW else (1 / (1 + np.exp(-xi)) if k in bounded else np.exp(xi))
    return v
out = np.zeros((len(tt), 2)); nT = len(tt)
XP = np.zeros((nT, 3)); PP = np.zeros((nT, 3, 3)); XF = np.zeros((nT, 3)); PF = np.zeros((nT, 3, 3)); AA = np.zeros((nT, 3, 3))
def nll(x):
    v = unpack(x)
    return -kf(tt, z, R, vv, sg, ff, reset, skip, np.array([v[k] for k in names], float), out, XP, PP, XF, PF, AA)


@njit(cache=True)
def rts(XP, PP, XF, PF, AA, XS, ff, sg, vv, th):
    # note: kick between XF[j] and XP[j+1] is deterministic -> add it back via difference of means
    n = XP.shape[0]; XS[n - 1] = XF[n - 1]; PS = PF[n - 1].copy()
    for j in range(n - 2, -1, -1):
        # post-kick state after obs j
        xk = XF[j].copy(); Pk = PF[j].copy()
        if ff[j] > 0:
            g = vv[j]; mu = th[0 + g] * sg[j] * ff[j] ** th[2 + g]; p = th[4 + g]; px = th[8 + g]; q = th[10 + g]
            xk[0] += p * mu; xk[1] += (1 - p) * mu; xk[2] += th[19 + g] * mu
            Pk[0, 0] += px * px * q; Pk[1, 1] += (1 - px) ** 2 * q; Pk[0, 1] += px * (1 - px) * q; Pk[1, 0] += px * (1 - px) * q
        A = AA[j + 1]
        dx = XS[j + 1] - XP[j + 1]
        XS[j] = XF[j] + (Pk @ A.T @ np.linalg.inv(PP[j + 1] + np.eye(3) * 1e-12)) @ dx
        # (covariance not needed)
    return XS
x0 = [fixed[k] if k in RAW else (np.log(fixed[k] / (1 - fixed[k])) if k in bounded else np.log(fixed[k])) for k in free]
res = optimize.minimize(nll, x0, method='Nelder-Mead', options={'maxiter': 40000, 'maxfev': 40000, 'xatol': 1e-4, 'fatol': 1e-3, 'adaptive': True})
res = optimize.minimize(nll, res.x, method='Nelder-Mead', options={'maxiter': 40000, 'maxfev': 40000, 'xatol': 1e-5, 'fatol': 1e-4, 'adaptive': True})
v = unpack(res.x)
nll(res.x)
print(tag, variant, 'nll %.1f' % res.fun, 'k', len(free), {k: round(float(v[k]), 5) for k in names}, flush=True)
json.dump(dict(variant=variant, nll=res.fun, params={k: float(v[k]) for k in names}), open(prep.common.CACHE / f'kfd_{tag}_{variant}.json', 'w'))
XS = np.zeros((nT, 3))
rts(XP, PP, XF, PF, AA, XS, ff, sg, vv, np.array([v[k] for k in names], float))
np.savez(prep.common.CACHE / f'kfd_{tag}_{variant}_state.npz', t=tt, filt=XF, smooth=XS, io=io)
