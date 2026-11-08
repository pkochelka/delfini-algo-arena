"""Velocity law test: grid HMM over the S velocity v, observed through binned residual increments
y_k = E(t_k+B) - E(t_k), E = S - deterministic kick impact (drift NOT removed). y ~ N(v B, R_k), R_k = q_reg * n_fills_k + r0.
Transition per bin, per regime (0 calm, 1 vol):
  OU     : v' = a v + N(0, s^2 (1 - a^2)),  a = exp(-B/Tm)
  RESET  : w.p. 1-exp(-B/Tm) v' ~ N(0, s^2) (fresh draw), else v' = v
  MIX    : OU + resets
  ONOFF  : (calm) zero state + Gaussian episodes: episodes start at rate lam_on to N(0, s^2) and end at rate 1/Tm to 0
Usage: vhmm.py TAG params.json MODEL [B]"""
import sys, json, numpy as np, pandas as pd
from numba import njit
from scipy import optimize
import prep, smooth_seg as SS
tag, pf, model = sys.argv[1], sys.argv[2], sys.argv[3]; B = float(sys.argv[4]) if len(sys.argv) > 4 else 0.25
P = json.load(open(prep.common.HERE / pf))
d = SS.d; r = dict(np.load(prep.common.CACHE / f'smooth_{tag}.npz'))
t = d['t']; m = r['mean']; vol = r['vol'].astype(bool)
def arr(k, dflt): return np.where(vol, P['vol'].get(k, dflt), P['calm'].get(k, dflt)).astype(float)
d['vshift'][:] = 0.0
shift = SS.impact(slice(0, len(t)), arr('c', 0), arr('e', 0.6), arr('pperm', 1.0), arr('tdec', 15.0))
E = m - np.cumsum(np.r_[0, shift[:-1]])
g = np.arange(0, 6000, B); i = np.clip(np.searchsorted(t, g, 'right') - 1, 0, len(t) - 1)
y = np.diff(E[i]); reg = vol[i[1:]].astype(np.int64)
nf = np.bincount(np.minimum((t[d['ioc'] & (d['filled'] > 0)] / B).astype(int), len(y) - 1), minlength=len(y)).astype(float)
skip = np.zeros(len(y), np.int64)
SH = pd.read_csv(prep.common.HERE / '../flow/shock_events.csv').onset.to_numpy()
for a in SH: skip[int((a - 0.5) / B):int((a + 6) / B) + 1] = 1
VG = np.arange(-15, 15.001, 0.1); NV = len(VG)


@njit(cache=True)
def run(y, reg, nf, skip, B, VG, th, model, store, post):
    # th: [Tm0, Tm1, s0, s1, q0, q1, r0, Tr0, Tr1, lon0, lon1]
    NV = len(VG); h = VG[1] - VG[0]
    p = np.zeros(NV)
    for j in range(NV): p[j] = np.exp(-0.5 * (VG[j] / th[3]) ** 2)
    p /= p.sum()
    tmp = np.zeros(NV); ll = 0.0
    K = np.zeros((2, NV, NV))
    for g in range(2):
        Tm = th[0 + g]; s = th[2 + g]; Tr = th[7 + g]; lon = th[9 + g]
        a = np.exp(-B / Tm)
        prior = np.zeros(NV)
        for j in range(NV): prior[j] = np.exp(-0.5 * (VG[j] / s) ** 2)
        prior /= prior.sum()
        for u in range(NV):          # from u
            row = np.zeros(NV)
            if model == 0 or model == 2:   # OU part
                sd = s * np.sqrt(max(1 - a * a, 1e-9)); mu = a * VG[u]
                if sd < h * 0.3:
                    sd = h * 0.3
                for w in range(NV): row[w] = np.exp(-0.5 * ((VG[w] - mu) / sd) ** 2)
                row /= row.sum()
            if model == 1:
                for w in range(NV): row[w] = 0.0
                row[u] = 1.0
            if model == 2 or model == 1:
                q = 1 - np.exp(-B / (Tr if model == 2 else Tm))
                for w in range(NV): row[w] = (1 - q) * row[w] + q * prior[w]
            if model == 3:
                # on/off: state 0 (v=0 bin) ; Tm = episode mean dur, lon = start rate
                z = NV // 2
                for w in range(NV): row[w] = 0.0
                if u == z:
                    q = 1 - np.exp(-B * lon)
                    for w in range(NV): row[w] = q * prior[w]
                    row[z] += 1 - q
                else:
                    q = 1 - np.exp(-B / Tm)
                    row[u] = 1 - q; row[z] += q
            for w in range(NV): K[g, u, w] = row[w]
    for k in range(len(y)):
        g = reg[k]
        tmp[:] = 0.0
        for u in range(NV):
            if p[u] > 1e-10:
                for w in range(NV): tmp[w] += p[u] * K[g, u, w]
        p[:] = tmp
        if not skip[k]:
            R = th[4 + g] * nf[k] + th[6]
            s = 0.0
            for w in range(NV):
                v = p[w] * np.exp(-0.5 * (y[k] - VG[w] * B) ** 2 / R) / np.sqrt(2 * np.pi * R)
                tmp[w] = v; s += v
            ll += np.log(s + 1e-300)
            p[:] = tmp / (s + 1e-300)
        if store:
            for w in range(NV): post[k, w] = p[w]
    return ll


MOD = dict(OU=0, RESET=1, MIX=2, ONOFF=3)[model]
names = ['Tm0', 'Tm1', 's0', 's1', 'q0', 'q1', 'r0', 'Tr0', 'Tr1', 'lon0', 'lon1']
base = dict(Tm0=1.0, Tm1=3.0, s0=0.4, s1=3.0, q0=0.01, q1=0.04, r0=0.003, Tr0=5.0, Tr1=5.0, lon0=0.1, lon1=0.5)
free = dict(OU=['Tm0', 'Tm1', 's0', 's1', 'q0', 'q1', 'r0'], RESET=['Tm0', 'Tm1', 's0', 's1', 'q0', 'q1', 'r0'],
            MIX=['Tm0', 'Tm1', 's0', 's1', 'q0', 'q1', 'r0', 'Tr0', 'Tr1'],
            ONOFF=['Tm0', 'Tm1', 's0', 's1', 'q0', 'q1', 'r0', 'lon0', 'lon1'])[model]
if model == 'ONOFF': base.update(Tm0=5.0, Tm1=3.0, s0=0.7, lon0=0.05, lon1=1.0)
dummy = np.zeros((1, 1))
def nll(x):
    v = dict(base); v.update(dict(zip(free, np.exp(x))))
    return -run(y, reg, nf, skip, B, VG, np.array([v[k] for k in names]), MOD, False, dummy)
x0 = np.log([base[k] for k in free])
res = optimize.minimize(nll, x0, method='Nelder-Mead', options={'maxiter': 3000, 'xatol': 1e-3, 'fatol': 0.05, 'adaptive': True})
v = dict(base); v.update(dict(zip(free, np.exp(res.x))))
print(tag, model, 'B', B, 'nll %.1f' % res.fun, 'k', len(free), {k: round(float(v[k]), 4) for k in free}, flush=True)
json.dump(dict(model=model, nll=res.fun, params=v), open(prep.common.CACHE / f'vhmm_{tag}_{model}.json', 'w'))
# profile one parameter (env PROF=name:v1,v2,...) around the fit
import os
if os.environ.get('PROF'):
    nm, vals = os.environ['PROF'].split(':')
    for val in map(float, vals.split(',')):
        w = dict(v); w[nm] = val
        print('  profile', nm, val, 'nll %.1f' % -run(y, reg, nf, skip, B, VG, np.array([w[k] for k in names]), MOD, False, dummy), flush=True)
