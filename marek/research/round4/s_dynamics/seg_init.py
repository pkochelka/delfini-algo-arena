"""Provisional regime segmentation: 2-state Viterbi on per-second Poisson counts of multi-rung ladder bursts and takers.
-> _cache/vol_segments_init.csv"""
import numpy as np, pandas as pd
import prep
d = prep.build(); t = d['t']
L = np.load(prep.common.CACHE / 'ladder_member.npy')
sec = np.floor(t).astype(int)
X = np.c_[np.bincount(sec[L[:, 1] == 1], minlength=6000), np.bincount(sec[d['ioc']], minlength=6000)].astype(float)
lab = (pd.Series(X[:, 0]).rolling(10, center=True, min_periods=1).mean() > 4).to_numpy().astype(int)
pen = 15.0
for it in range(30):
    lam = np.array([X[lab == s].mean(0) for s in (0, 1)])
    ll = np.stack([(X * np.log(lam[s]) - lam[s]).sum(1) for s in (0, 1)], 1)
    n = len(X); V = np.zeros((n, 2)); B = np.zeros((n, 2), int); V[0] = ll[0]
    for i in range(1, n):
        for s in (0, 1):
            c = [V[i - 1, 0] - (pen if s != 0 else 0), V[i - 1, 1] - (pen if s != 1 else 0)]
            B[i, s] = int(np.argmax(c)); V[i, s] = max(c) + ll[i, s]
    new = np.zeros(n, int); new[-1] = int(np.argmax(V[-1]))
    for i in range(n - 1, 0, -1): new[i - 1] = B[i, new[i]]
    if (new == lab).all(): break
    lab = new
print('rates (bursts/s, takers/s) calm, vol', lam.round(2))
ch = np.flatnonzero(np.diff(lab)) + 1
edges = np.r_[0, ch, len(lab)]
R = pd.DataFrame([(a, b, lab[a]) for a, b in zip(edges[:-1], edges[1:])], columns=['start', 'end', 'high'])
R['dur'] = R.end - R.start
V = R[R.high == 1][['start', 'end', 'dur']]
print(V.to_string())
print('n vol', len(V), 'frac', V.dur.sum() / 6000, 'mean vol dur', V.dur.mean(), 'mean calm dur', R[R.high == 0].dur.mean())
V.to_csv(prep.common.CACHE / 'vol_segments_init.csv', index=False)
