"""Distributed-lag (propagator) estimate: dS over bin t = sum_L g_reg(L) * Q_reg(t-L), Q = sum sign*filled^e in bin,
separately for takers in calm / vol. Cumulative G(L) = response of S at lag L to unit sign*f^e. Path TAG."""
import sys, numpy as np, pandas as pd
import prep
tag = sys.argv[1] if len(sys.argv) > 1 else 'np'
here = prep.common.HERE
B = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
LMAX = int(sys.argv[3]) if len(sys.argv) > 3 else 150
e = 0.6
shf = sys.argv[4] if len(sys.argv) > 4 else str(here / '../flow/shock_events.csv')
d = prep.build(); r = dict(np.load(prep.common.CACHE / f'smooth_{tag}.npz'))
t = d['t']; m = r['mean']; vol = r['vol'].astype(bool)
nb = int(6000 / B); edges = np.arange(nb + 1) * B
i = np.clip(np.searchsorted(t, edges, 'right') - 1, 0, len(t) - 1)
S = m[i]; y = np.diff(S)
tk = d['ioc'] & (d['filled'] > 0)
q = np.where(tk, d['sign'] * d['filled'] ** e, 0.0)
bi = np.minimum((t / B).astype(int), nb - 1)
Qc = np.bincount(bi, q * ~vol, nb); Qv = np.bincount(bi, q * vol, nb)
SH = pd.read_csv(shf).onset.to_numpy()
bad = np.zeros(nb, bool)
for a in SH: bad[max(0, int((a + 1.5) / B)):int((a + 5.0) / B) + 1] = True
L = int(LMAX / B)
rows = np.arange(L, nb)
rows = rows[~bad[rows]]
X = np.zeros((len(rows), 2 * (L + 1)))
for l in range(L + 1):
    X[:, l] = Qc[rows - l]; X[:, L + 1 + l] = Qv[rows - l]
Y = y[rows]
b, *_ = np.linalg.lstsq(X, Y, rcond=None)
res = Y - X @ b
cov = np.linalg.inv(X.T @ X) * res.var()
gc, gv = b[:L + 1], b[L + 1:]
Gc, Gv = np.cumsum(gc), np.cumsum(gv)
sc = np.sqrt([cov[:k + 1, :k + 1].sum() for k in range(L + 1)])
sv = np.sqrt([cov[L + 1:L + 2 + k, L + 1:L + 2 + k].sum() for k in range(L + 1)])
print('bin %.2f s, n %d, resid var %.4f' % (B, len(rows), res.var()))
for lag in [0, 1, 2, 3, 5, 10, 15, 20, 30, 45, 60, 90, 120, 150, 200, 300]:
    k = int(lag / B)
    if k <= L:
        print('  lag %5.1f s  G_calm %.4f (%.4f)   G_vol %.4f (%.4f)' % (lag, Gc[k], sc[k], Gv[k], sv[k]))
np.savez(prep.common.CACHE / f'prop_{tag}_{B}.npz', Gc=Gc, Gv=Gv, sc=sc, sv=sv)
