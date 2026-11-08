"""Multi-stream Poisson HMM on 1 s bins of round 3. Streams: takers, ladder links (exact 2000 us), maker inserts,
maker inserts of 20-70 lots, cancels. Fit K = 2..5 by Baum-Welch (log space), report BIC, state means, dwell,
and agreement with the s_dynamics volatile segments."""
import numpy as np, pandas as pd, json, sys
from scipy.special import gammaln, logsumexp
df = pd.read_csv('data/round3.csv')
t = df.Time.values; b = np.floor(t).astype(int); nb = b.max() + 1
ins = df.Operation.values == 'Insert'; ioc = df.Lifespan.values == 'F'
mk = ins & ~ioc
side = df.Side.values; px = df.Price.values; us = np.round(t * 1e6).astype(np.int64)
lad = np.zeros(len(df), bool)
for sd, d in [('B', -1), ('A', 1)]:
    idx = np.nonzero(mk & (side == sd))[0]
    gap = np.diff(us[idx]); dp = np.diff(px[idx])
    lad[idx[1:][(np.abs(gap - 2000) <= 3) & (dp == d)]] = True
vol = df.Volume.values
streams = {
    'takers': ins & ioc,
    'ladder': lad,
    'makers': mk & ~lad,
    'mk20_70': mk & ~lad & (vol >= 20) & (vol <= 70),
    'cancels': df.Operation.values == 'Cancel',
}
X = np.stack([np.bincount(b[m], minlength=nb) for m in streams.values()], 1).astype(float)
names = list(streams)
np.save('marek/research/round3/hmm/X1s.npy', X)

def fit(X, K, iters=200, seed=0):
    rng = np.random.default_rng(seed)
    T, D = X.shape
    q = np.quantile(X, np.linspace(0.1, 0.9, K), axis=0)
    lam = q + rng.uniform(0, 1, q.shape)
    A = np.full((K, K), 0.02 / max(K - 1, 1)); np.fill_diagonal(A, 0.98)
    pi = np.full(K, 1 / K)
    lgX = gammaln(X + 1).sum(1)
    prev = -np.inf
    for it in range(iters):
        logB = X @ np.log(lam).T - lam.sum(1) - lgX[:, None]
        la = np.empty((T, K)); lA = np.log(A)
        la[0] = np.log(pi) + logB[0]
        for i in range(1, T):
            la[i] = logsumexp(la[i - 1][:, None] + lA, 0) + logB[i]
        lb = np.zeros((T, K))
        for i in range(T - 2, -1, -1):
            lb[i] = logsumexp(lA + logB[i + 1] + lb[i + 1], 1)
        ll = logsumexp(la[-1])
        g = np.exp(la + lb - ll)
        xi = np.zeros((K, K))
        for i in range(T - 1):
            m = la[i][:, None] + lA + logB[i + 1] + lb[i + 1] - ll
            xi += np.exp(m)
        A = xi / xi.sum(1, keepdims=True); pi = g[0] + 1e-9; pi /= pi.sum()
        lam = (g.T @ X) / g.sum(0)[:, None] + 1e-6
        if ll - prev < 1e-3: break
        prev = ll
    return ll, lam, A, g
vs = pd.read_csv('marek/research/round3/s_dynamics/vol_segments.csv')
truth = np.zeros(nb, bool)
for a, c in zip(vs.start, vs.end): truth[int(a):int(np.ceil(c))] = True
sh = pd.read_csv('marek/research/round3/flow/shock_events.csv')
dr = np.zeros(nb, bool)
for o in sh.onset: dr[int(o):int(o) + 4] = True
out = {}
for K in [2, 3, 4, 5]:
    best = max((fit(X, K, seed=s) for s in range(3)), key=lambda r: r[0])
    ll, lam, A, g = best
    T, D = X.shape
    npar = K * D + K * (K - 1) + (K - 1)
    bic = -2 * ll + npar * np.log(T)
    st = g.argmax(1)
    print(f'\nK={K} ll={ll:.0f} BIC={bic:.0f}')
    for k in range(K):
        m = st == k
        print(f'  state {k}: share {m.mean():.3f} dwell {1/(1-A[k,k]):6.1f}s  P(vol|s) {truth[m].mean():.2f}  P(drought|s) {dr[m].mean():.2f}  '
              + ' '.join(f'{n}={v:.1f}' for n, v in zip(names, lam[k])))
    out[K] = dict(ll=ll, bic=bic, lam=lam.tolist(), A=A.tolist())
json.dump(out, open('marek/research/round3/hmm/hmm_fit.json', 'w'), indent=1)
