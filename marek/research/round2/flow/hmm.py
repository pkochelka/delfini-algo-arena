"""Poisson HMM on taker counts (buy+sell) in 1 s bins, shock windows treated as missing. K = 1..5, BIC."""
import numpy as np
from pathlib import Path
from scipy.special import gammaln
D = Path(__file__).parent; C = D / 'cache'
z = np.load(C / 'bins1s.npz'); x = z['tk']; mask = z['mask']; n = len(x)

def fit(K, iters=300, seed=0):
    rng = np.random.default_rng(seed)
    lam = np.quantile(x[mask], np.linspace(0.15, 0.9, K)) + rng.uniform(0, .5, K)
    A = np.full((K, K), 0.01 / max(K - 1, 1)); np.fill_diagonal(A, 0.99) if K > 1 else None
    if K == 1: A = np.ones((1, 1))
    pi = np.ones(K) / K; prev = -np.inf
    for it in range(iters):
        logB = x[:, None] * np.log(lam)[None] - lam[None] - gammaln(x + 1)[:, None]
        logB[~mask] = 0
        B = np.exp(logB - logB.max(1, keepdims=True)); off = logB.max(1)
        al = np.zeros((n, K)); c = np.zeros(n)
        a = pi * B[0]; c[0] = a.sum(); al[0] = a / c[0]
        for t in range(1, n):
            a = (al[t - 1] @ A) * B[t]; c[t] = a.sum(); al[t] = a / c[t]
        ll = np.log(c).sum() + off.sum()
        be = np.ones((n, K))
        for t in range(n - 2, -1, -1):
            b = A @ (B[t + 1] * be[t + 1]); be[t] = b / c[t + 1]
        g = al * be; g /= g.sum(1, keepdims=True)
        xi = np.zeros((K, K))
        for t in range(n - 1):
            m = al[t][:, None] * A * (B[t + 1] * be[t + 1])[None] / c[t + 1]; xi += m
        A = xi / xi.sum(1, keepdims=True); pi = g[0]
        lam = (g[mask] * x[mask, None]).sum(0) / g[mask].sum(0)
        if ll - prev < 1e-4: break
        prev = ll
    o = np.argsort(lam)
    return ll, lam[o], A[np.ix_(o, o)], g[:, o]
res = {}
for K in [1, 2, 3, 4, 5]:
    ll, lam, A, g = fit(K)
    k = K + K * (K - 1)
    bic = -2 * ll + k * np.log(mask.sum())
    res[K] = (ll, lam, A, g)
    print(f'K={K} ll={ll:.1f} BIC={bic:.1f}')
    print('  rates', np.round(lam, 3))
    if K > 1:
        print('  mean dwell (s)', np.round(1 / (1 - np.diag(A)), 1))
        print('  occupancy', np.round(g[mask].mean(0), 3))
        print('  A\n', np.round(A, 4))
np.save(C / 'hmm_taker_gamma3.npy', res[3][3]); np.save(C / 'hmm_taker_gamma2.npy', res[2][3]); np.save(C / 'hmm_taker_gamma4.npy', res[4][3])
