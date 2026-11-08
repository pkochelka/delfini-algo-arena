"""Does the per-fill noise xi decay? Propagator regression of dS on lagged Q (signed filled^e, calm/vol) and lagged
K = smoothed kick residual at each filled taker (S after - S before - modelled kick). Path TAG, params JSON."""
import sys, json, numpy as np, pandas as pd
import prep
tag, pf = sys.argv[1], sys.argv[2]
B = 1.0; LMAX = 120
P = json.load(open(prep.common.HERE / pf))
d = prep.build(); r = dict(np.load(prep.common.CACHE / f'smooth_{tag}.npz'))
t = d['t']; m = r['mean']; vol = r['vol'].astype(bool)
c = np.where(vol, P['vol']['c'], P['calm']['c']); e = np.where(vol, P['vol']['e'], P['calm']['e'])
tk = d['ioc'] & (d['filled'] > 0)
kick = np.where(tk, c * d['sign'] * d['filled'] ** e, 0.0)
K = np.zeros(len(t)); K[:-1] = np.where(tk[:-1], m[1:] - m[:-1] - kick[:-1], 0.0)
nb = int(6000 / B); edges = np.arange(nb + 1) * B
i = np.clip(np.searchsorted(t, edges, 'right') - 1, 0, len(t) - 1)
y = np.diff(m[i])
bi = np.minimum((t / B).astype(int), nb - 1)
q = np.where(tk, d['sign'] * d['filled'] ** e, 0.0)
regs = [np.bincount(bi, q * ~vol, nb), np.bincount(bi, q * vol, nb), np.bincount(bi, K * ~vol, nb), np.bincount(bi, K * vol, nb)]
SH = pd.read_csv(prep.common.HERE / '../flow/shock_events.csv').onset.to_numpy()
bad = np.zeros(nb, bool)
for a in SH: bad[max(0, int((a + 1.5) / B)):int((a + 5.0) / B) + 1] = True
L = int(LMAX / B); rows = np.arange(L, nb); rows = rows[~bad[rows]]
X = np.zeros((len(rows), 4 * (L + 1)))
for g, Q in enumerate(regs):
    for l in range(L + 1):
        X[:, g * (L + 1) + l] = Q[rows - l]
b, *_ = np.linalg.lstsq(X, y[rows], rcond=None)
res = y[rows] - X @ b; cov = np.linalg.inv(X.T @ X) * res.var()
names = ['Q_calm', 'Q_vol', 'K_calm', 'K_vol']
for g, nm in enumerate(names):
    G = np.cumsum(b[g * (L + 1):(g + 1) * (L + 1)])
    se = [np.sqrt(cov[g * (L + 1):g * (L + 1) + k + 1, g * (L + 1):g * (L + 1) + k + 1].sum()) for k in range(L + 1)]
    print(nm, ' '.join('%d:%.4f(%.4f)' % (lag, G[lag], se[lag]) for lag in [0, 2, 5, 10, 20, 30, 60, 90, 120]))
