"""2/3-state Poisson HMM on 1 s maker counts by size band (<=10, 20-70, 90-320, rest), shock windows masked.
Saves cache/regime_sz.npz: post (n_sec x K), lam, A; and cache/regime_sz.csv with per-second state + segments."""
import numpy as np, pandas as pd
from pathlib import Path
from scipy.special import gammaln
from prep import load
D = Path(__file__).parent; C = D / 'cache'
o = load(); m = o[~o.ioc.astype(bool)]
sec = (m.t_us // 1_000_000).to_numpy(); v = m.volume.to_numpy()
band = np.select([v <= 10, (v >= 20) & (v <= 70), (v >= 90) & (v <= 320)], [0, 1, 2], 3)
n = 6000
X = np.zeros((n, 4)); np.add.at(X, (sec, band), 1)
ev = pd.read_csv(D.parent / 'flow' / 'shock_events.csv')
mask = np.ones(n, bool)
for t in ev.onset: mask[max(0, int(t) - 1):int(t) + 8] = False

def fit(X, K, iters=300):
    lam = np.stack([np.quantile(X[mask, j], np.linspace(0.3, 0.8, K)) for j in range(X.shape[1])], 1) + 0.5
    lam[:, 2] = lam[::-1, 2]
    A = np.full((K, K), 0.01 / max(K - 1, 1)); np.fill_diagonal(A, 0.99); pi = np.ones(K) / K; prev = -np.inf
    lg = gammaln(X + 1).sum(1)
    for it in range(iters):
        logB = X @ np.log(lam).T - lam.sum(1)[None] - lg[:, None]; logB[~mask] = 0
        off = logB.max(1); B = np.exp(logB - off[:, None])
        al = np.zeros((n, K)); c = np.zeros(n); a = pi * B[0]; c[0] = a.sum(); al[0] = a / c[0]
        for t in range(1, n):
            a = (al[t - 1] @ A) * B[t]; c[t] = a.sum(); al[t] = a / c[t]
        ll = np.log(c).sum() + off.sum(); be = np.ones((n, K))
        for t in range(n - 2, -1, -1): be[t] = A @ (B[t + 1] * be[t + 1]) / c[t + 1]
        g = al * be; g /= g.sum(1, keepdims=True)
        xi = np.einsum('ti,ij,tj->ij', al[:-1], A, B[1:] * be[1:] / c[1:, None])
        A = xi / xi.sum(1, keepdims=True); pi = g[0]
        lam = (g[mask].T @ X[mask]) / g[mask].sum(0)[:, None]
        if ll - prev < 1e-6: break
        prev = ll
    o_ = np.argsort(lam[:, 1]); return ll, lam[o_], A[np.ix_(o_, o_)], g[:, o_]

res = {}
for K in (2, 3):
    ll, lam, A, g = fit(X, K)
    k = K * 4 + K * (K - 1)
    print(f'K={K} ll={ll:.1f} BIC={-2*ll + k*np.log(mask.sum()):.1f}')
    print(' rates (<=10, 20-70, 90-320, rest):', np.round(lam, 2).tolist())
    print(' dwell s', np.round(1 / (1 - np.diag(A)), 1), 'occ', np.round(g[mask].mean(0), 3))
    res[K] = (lam, A, g)
lam, A, g = res[2]
st = (g[:, 1] > 0.5).astype(int)
np.savez(C / 'regime_sz.npz', post=g, lam=lam, A=A)
# segments
chg = np.flatnonzero(np.diff(st)) + 1
starts = np.r_[0, chg]; ends = np.r_[chg, n]
seg = pd.DataFrame({'start': starts, 'end': ends, 'vol': st[starts]}); seg['dur'] = seg.end - seg.start
seg.to_csv(C / 'regime_sz_segments.csv', index=False)
print(seg[seg.vol == 1].dur.describe()); print(seg[seg.vol == 0].dur.describe())
print('vol share', st.mean(), 'n vol segs', (seg.vol == 1).sum())
print('uncertain seconds (0.1<p<0.9):', ((g[:, 1] > .1) & (g[:, 1] < .9)).sum())
