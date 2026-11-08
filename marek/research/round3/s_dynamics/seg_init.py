"""Provisional regime segmentation from per-second observables (2-state Viterbi, Gaussian emissions on features)."""
import numpy as np, pandas as pd
import prep
d = prep.build(); t = d['t']; k = d['kind']
sec = np.floor(t).astype(int)
sp = d['ba'] - d['bb']
f = pd.DataFrame(dict(sec=sec, mk=(k <= 1), cl=(k == 2) | (k == 3), tk=(k >= 4), sp=sp))
g = f.groupby('sec').agg(mk=('mk', 'sum'), cl=('cl', 'sum'), tk=('tk', 'sum'), sp=('sp', 'median'))
print(np.histogram(g.mk, bins=30))
print(np.histogram(g.cl, bins=30))
X = np.c_[g.mk, g.cl, g.tk].astype(float)
# init with threshold
lab = (g.mk > 140).to_numpy().astype(int)
for it in range(10):
    mu = np.array([X[lab == s].mean(0) for s in (0, 1)]); sd = np.array([X[lab == s].std(0) for s in (0, 1)])
    ll = np.stack([(-0.5 * ((X - mu[s]) / sd[s]) ** 2 - np.log(sd[s])).sum(1) for s in (0, 1)], 1)
    pen = 15.0
    n = len(X); V = np.zeros((n, 2)); B = np.zeros((n, 2), int)
    V[0] = ll[0]
    for i in range(1, n):
        for s in (0, 1):
            c = [V[i - 1, 0] - (pen if s != 0 else 0), V[i - 1, 1] - (pen if s != 1 else 0)]
            B[i, s] = int(np.argmax(c)); V[i, s] = max(c) + ll[i, s]
    new = np.zeros(n, int); new[-1] = int(np.argmax(V[-1]))
    for i in range(n - 1, 0, -1): new[i - 1] = B[i, new[i]]
    if (new == lab).all(): break
    lab = new
print('means', mu.round(1), 'sd', sd.round(1))
s = g.index.to_numpy()
ch = np.flatnonzero(np.diff(lab)) + 1
edges = np.r_[0, ch, len(lab)]
rows = [(s[a], s[b - 1] + 1, lab[a]) for a, b in zip(edges[:-1], edges[1:])]
R = pd.DataFrame(rows, columns=['start', 'end', 'high'])
R['dur'] = R.end - R.start
print(R.to_string())
V = R[R.high == 1][['start', 'end']]
print('n vol', len(V), 'frac', (V.end - V.start).sum() / 6000, 'mean vol dur', (V.end - V.start).mean(), 'mean calm dur', R[R.high == 0].dur.mean())
V.to_csv(prep.common.HERE / '_cache/vol_segments_init.csv', index=False)
