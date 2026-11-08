"""Poisson HMMs on 1 s counts (shock windows -2..+20 s missing): taker-only, maker-only, joint. K=1..4, BIC.
Saves posteriors to cache/hmm_{tk,mk,joint}_K.npy."""
import numpy as np, pandas as pd, sys
from pathlib import Path
from scipy.special import gammaln
D = Path(__file__).parent; C = D / 'cache'
z = np.load(C / 'bins.npz'); ev = pd.read_csv(D / 'shock_events.csv')
tk = z['tb1'] + z['ts1']; mk = z['mb1'] + z['ma1']; n = len(tk)
mask = np.ones(n, bool)
for o in ev.onset: mask[max(0, int(o) - 2):int(o) + 20] = False
np.savez(C / 'bins1s.npz', tk=tk, mk=mk, mask=mask)

def fit(X, K, iters=500, seed=0):
    rng = np.random.default_rng(seed); d = X.shape[1]
    lam = np.stack([np.quantile(X[mask, j], np.linspace(0.2, 0.9, K)) for j in range(d)], 1) * (1 + rng.uniform(0, .01, (K, d)))
    A = np.full((K, K), 0.01 / max(K - 1, 1)); np.fill_diagonal(A, 0.99)
    if K == 1: A = np.ones((1, 1))
    pi = np.ones(K) / K; prev = -np.inf
    lg = gammaln(X + 1).sum(1)
    for it in range(iters):
        logB = X @ np.log(lam).T - lam.sum(1)[None] - lg[:, None]
        logB[~mask] = 0
        off = logB.max(1); B = np.exp(logB - off[:, None])
        al = np.zeros((n, K)); c = np.zeros(n)
        a = pi * B[0]; c[0] = a.sum(); al[0] = a / c[0]
        for t in range(1, n):
            a = (al[t - 1] @ A) * B[t]; c[t] = a.sum(); al[t] = a / c[t]
        ll = np.log(c).sum() + off.sum()
        be = np.ones((n, K))
        for t in range(n - 2, -1, -1):
            be[t] = A @ (B[t + 1] * be[t + 1]) / c[t + 1]
        g = al * be; g /= g.sum(1, keepdims=True)
        xi = np.einsum('ti,ij,tj->ij', al[:-1], A, B[1:] * be[1:] / c[1:, None])
        A = xi / xi.sum(1, keepdims=True); pi = g[0]
        lam = (g[mask].T @ X[mask]) / g[mask].sum(0)[:, None]
        if ll - prev < 1e-5: break
        prev = ll
    o = np.argsort(lam[:, 0])
    return ll, lam[o], A[np.ix_(o, o)], g[:, o]

for name, X in [('tk', tk[:, None]), ('mk', mk[:, None]), ('joint', np.c_[tk, mk])]:
    print('=====', name)
    for K in [1, 2, 3, 4]:
        ll, lam, A, g = fit(X.astype(float), K)
        k = K * X.shape[1] + K * (K - 1)
        print(f'K={K} ll={ll:.1f} BIC={-2 * ll + k * np.log(mask.sum()):.1f}')
        print('  rates', np.round(lam, 2).tolist())
        if K > 1:
            print('  dwell', np.round(1 / (1 - np.diag(A)), 1), 'occ', np.round(g[mask].mean(0), 3))
            print('  A', np.round(A, 4).tolist())
        np.save(C / f'hmm_{name}_{K}.npy', g)
