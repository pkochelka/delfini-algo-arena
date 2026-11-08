"""Is the super-diffusion of S impact from persistent flow, or a drift of S's own?
1 s grid; shock bins (onset-0.5 .. onset+6) zeroed. dS = S increments, dI = deterministic impact (model kicks + decay),
dE = dS - dI. Variance ratios VR(h) = Var(sum_h)/ (h Var(1 s)), ACF of dE, cross-corr of dE with flow.
Usage: drift.py TAG params.json"""
import sys, json, numpy as np, pandas as pd
import prep, smooth_seg as SS
tag, pf = sys.argv[1], sys.argv[2]
P = json.load(open(prep.common.HERE / pf))
d = SS.d; r = dict(np.load(prep.common.CACHE / f'smooth_{tag}.npz'))
t = d['t']; m = r['mean']; vol = r['vol'].astype(bool)
def arr(k, dflt): return np.where(vol, P['vol'].get(k, dflt), P['calm'].get(k, dflt)).astype(float)
shift = SS.impact(slice(0, len(t)), arr('c', 0), arr('e', 0.6), arr('pperm', 1.0), arr('tdec', 15.0))
I = np.cumsum(np.r_[0, shift[:-1]])
B = 1.0
g = np.arange(0, 6000 + B, B); i = np.clip(np.searchsorted(t, g, 'right') - 1, 0, len(t) - 1)
dS = np.diff(m[i]); dI = np.diff(I[i]); vg = vol[i[:-1]]
SH = pd.read_csv(prep.common.HERE / '../flow/shock_events.csv').onset.to_numpy()
bad = np.zeros(len(dS), bool)
for a in SH: bad[int((a - 0.5) / B):int((a + 6) / B) + 1] = True
dS[bad] = 0; dI[bad] = 0
dE = dS - dI
tk = d['ioc'] & (d['filled'] > 0)
bi = np.minimum((t / B).astype(int), len(dS) - 1)
Q = np.bincount(bi, np.where(tk, d['sign'] * d['filled'] ** 0.62, 0), len(dS))
N = np.bincount(bi, np.where(d['ioc'], d['sign'], 0), len(dS))
def vr(x, hs):
    v1 = x[~bad].var()
    out = []
    for h in hs:
        k = int(h / B); c = np.cumsum(np.r_[0, x]); s = c[k:] - c[:-k]
        out.append(s.var() / (k * v1))
    return np.array(out)
hs = [1, 2, 5, 10, 20, 30, 60, 120, 180, 300, 600]
print('h        :', ' '.join('%6g' % h for h in hs))
for nm, x in (('S', dS), ('impact', dI), ('resid E', dE)):
    v = vr(x, hs)
    print('%-9s:' % nm, ' '.join('%6.2f' % a for a in v), '  sd1 %.3f  sd300 %.1f' % (x[~bad].std(), np.sqrt(v[hs.index(300)] * 300) * x[~bad].std()))
print('var share at 300 s: var(sum dI)=%.0f var(sum dE)=%.0f cov=%.0f var(sum dS)=%.0f' % tuple(
    [np.var(np.convolve(x, np.ones(300), 'valid')) for x in (dI, dE)] + [np.cov(np.convolve(dI, np.ones(300), 'valid'), np.convolve(dE, np.ones(300), 'valid'))[0, 1], np.var(np.convolve(dS, np.ones(300), 'valid'))]))
def acf(x, y, L):
    x = x - x.mean(); y = y - y.mean()
    return np.array([np.corrcoef(x[:len(x) - l], y[l:])[0, 1] for l in L])
L = [1, 2, 3, 5, 10, 20, 30, 60]
print('lags     :', L)
for reg, msk in (('all', np.ones(len(dS), bool)),):
    print('ACF dS   ', acf(dS, dS, L).round(3))
    print('ACF dE   ', acf(dE, dE, L).round(3))
    print('ACF dI   ', acf(dI, dI, L).round(3))
    print('ACF Q    ', acf(Q, Q, L).round(3))
    print('corr(Q_t, dE_t+l)  ', acf(Q, dE, L).round(3), ' l=0: %.3f' % np.corrcoef(Q, dE)[0, 1])
    print('corr(dE_t, Q_t+l)  ', acf(dE, Q, L).round(3))
    print('corr(dS_t, Q_t+l)  ', acf(dS, Q, L).round(3))
    print('corr(dE_t, N_t+l)  ', acf(dE, N, L).round(3))
# by regime: dE variance per second
print('var dE /s: calm %.4f vol %.4f ; var dI /s calm %.4f vol %.4f' % (dE[~vg & ~bad].var(), dE[vg & ~bad].var(), dI[~vg & ~bad].var(), dI[vg & ~bad].var()))
np.savez(prep.common.CACHE / f'drift_{tag}.npz', dS=dS, dI=dI, dE=dE, Q=Q, N=N, vg=vg, bad=bad)
