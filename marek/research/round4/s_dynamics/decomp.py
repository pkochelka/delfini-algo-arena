"""Decompose S moves over horizon h (1 s grid, shocks removed) into drift integral (KF-smoothed velocity), deterministic
kick impact (with decay), and the rest (xi + error). Usage: decomp.py TAG params.json KFSTATE.npz"""
import sys, json, numpy as np, pandas as pd
import prep, smooth_seg as SS
tag, pf, kfs = sys.argv[1], sys.argv[2], sys.argv[3]
P = json.load(open(prep.common.HERE / pf))
d = SS.d; r = dict(np.load(prep.common.CACHE / f'smooth_{tag}.npz'))
t = d['t']; m = r['mean']; vol = r['vol'].astype(bool)
def arr(k, dflt): return np.where(vol, P['vol'].get(k, dflt), P['calm'].get(k, dflt)).astype(float)
d['vshift'][:] = 0.0
shift = SS.impact(slice(0, len(t)), arr('c', 0), arr('e', 0.6), arr('pperm', 1.0), arr('tdec', 15.0))
I = np.cumsum(np.r_[0, shift[:-1]])
z = np.load(kfs); v = np.interp(t, z['t'], z['smooth'][:, 2])
Dv = np.cumsum(np.r_[0, (v * np.r_[np.diff(t), 0])[:-1]])
g = np.arange(0, 6001, 1.0); i = np.clip(np.searchsorted(t, g, 'right') - 1, 0, len(t) - 1)
dS, dI, dD = np.diff(m[i]), np.diff(I[i]), np.diff(Dv[i]); vg = vol[i[:-1]]
SH = pd.read_csv(prep.common.HERE / '../flow/shock_events.csv').onset.to_numpy()
bad = np.zeros(len(dS), bool)
for a in SH: bad[int(a - 0.5):int(a + 6) + 1] = True
for x in (dS, dI, dD): x[bad] = 0
dX = dS - dI - dD
print('h      var(S)   var(drift) var(impact) var(rest)  2cov(D,I) 2cov(D,X) 2cov(I,X)  | VR(S)  sd(S)')
v1 = dS[~bad].var()
for h in (1, 2, 5, 10, 30, 60, 120, 300, 600):
    s = lambda x: np.convolve(x, np.ones(h), 'valid')
    S_, D_, I_, X_ = s(dS), s(dD), s(dI), s(dX)
    C = np.cov(np.vstack([D_, I_, X_]))
    print('%4d %9.1f %9.1f %9.1f %9.1f %9.1f %9.1f %9.1f   | %5.2f %6.1f' % (h, S_.var(), C[0, 0], C[1, 1], C[2, 2], 2 * C[0, 1], 2 * C[0, 2], 2 * C[1, 2], S_.var() / (h * v1), S_.std()))
for nm, msk in (('calm', ~vg & ~bad), ('vol', vg & ~bad)):
    print(nm, 'per-second var: S %.3f drift %.3f impact %.3f rest %.3f; corr(drift, impact) %.3f' % (
        dS[msk].var(), dD[msk].var(), dI[msk].var(), dX[msk].var(), np.corrcoef(dD[msk], dI[msk])[0, 1]))
# does past velocity predict flow impact? (feedback) and does past impact predict velocity?
for lag in (0, 1, 2, 5):
    a = dD[:len(dD) - lag] if lag else dD; b = dI[lag:]
    print('corr(drift_t, impact_t+%d) %.3f   corr(impact_t, drift_t+%d) %.3f' % (lag, np.corrcoef(a, b)[0, 1], lag, np.corrcoef(dI[:len(dI) - lag] if lag else dI, dD[lag:])[0, 1]))
