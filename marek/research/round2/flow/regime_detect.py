"""Regime detectability from the book: per-second mean spread and maker-insert envelope width vs HMM taker regime.
Then a sharper segmentation from a 2-state Gaussian HMM on per-second log maker-envelope width."""
import numpy as np, pandas as pd
from pathlib import Path
D = Path(__file__).parent; C = D / 'cache'
r = pd.read_pickle(C / 'rows.pkl'); hi = np.load(C / 'hmm_taker_gamma2.npy')[:, 1] > 0.5
z = np.load(C / 'bins1s.npz'); mask = z['mask']
spread = (r.ask_after - r.bid_after).to_numpy(); tt = r.t.to_numpy()
dt = np.diff(tt, append=6000.0)
sec = tt.astype(int).clip(0, 5999)
ok = np.isfinite(spread)
sp1 = np.bincount(sec[ok], spread[ok] * dt[ok], 6000) / np.bincount(sec[ok], dt[ok], 6000)
ins = r[(r.op == 0) & ~r.ioc].copy(); ins['b'] = (ins.t / 0.025).astype(int)
env = pd.DataFrame({'mb': ins[ins.side == 0].groupby('b').px.max(), 'ma': ins[ins.side == 1].groupby('b').px.min()}).dropna()
env['W'] = env.ma - env.mb; env['s'] = (env.index * 0.025).astype(int)
W1 = env.groupby('s').W.mean().reindex(range(6000)).to_numpy()
for name, x in [('spread', sp1), ('envW', W1)]:
    print(f'{name}: calm mean {np.nanmean(x[~hi & mask]):.3f} sd {np.nanstd(x[~hi & mask]):.3f} | volatile mean {np.nanmean(x[hi & mask]):.3f} sd {np.nanstd(x[hi & mask]):.3f}')
    for thr in [1.3, 1.5, 2.0, 2.5]:
        pred = x > thr; m = mask & np.isfinite(x)
        print(f'   thr {thr}: acc {np.mean(pred[m] == hi[m]):.3f}  TPR {pred[m & hi].mean():.3f} FPR {pred[m & ~hi].mean():.3f}')
print('hist envW per second (calm / volatile):')
bins = np.arange(0.75, 6.01, 0.25)
print(np.round(bins[:-1], 2)); print(np.histogram(W1[~hi & mask & np.isfinite(W1)], bins)[0]); print(np.histogram(W1[hi & mask & np.isfinite(W1)], bins)[0])
# 2-state Gaussian HMM on per-second envW (all seconds incl. shocks), simple EM
x = pd.Series(W1).interpolate().bfill().to_numpy(); n = len(x)
mu = np.array([1.6, 3.2]); sd = np.array([0.3, 0.8]); A = np.array([[0.995, 0.005], [0.03, 0.97]]); pi = np.array([0.9, 0.1])
for it in range(200):
    B = np.exp(-0.5 * ((x[:, None] - mu) / sd) ** 2) / sd
    al = np.zeros((n, 2)); c = np.zeros(n); a = pi * B[0]; c[0] = a.sum(); al[0] = a / c[0]
    for t in range(1, n): a = (al[t - 1] @ A) * B[t]; c[t] = a.sum(); al[t] = a / c[t]
    be = np.ones((n, 2))
    for t in range(n - 2, -1, -1): be[t] = A @ (B[t + 1] * be[t + 1]) / c[t + 1]
    g = al * be; g /= g.sum(1, keepdims=True)
    xi = sum(al[t][:, None] * A * (B[t + 1] * be[t + 1])[None] / c[t + 1] for t in range(n - 1))
    A = xi / xi.sum(1, keepdims=True); pi = g[0]
    mu = (g * x[:, None]).sum(0) / g.sum(0); sd = np.sqrt((g * (x[:, None] - mu) ** 2).sum(0) / g.sum(0))
print('Gaussian HMM on envW: mu', mu.round(3), 'sd', sd.round(3), 'A', A.round(5).tolist(), 'dwell', (1 / (1 - np.diag(A))).round(1))
h2 = g[:, 1] > 0.5
print('agreement with taker-count HMM (masked secs):', np.mean(h2[mask] == hi[mask]).round(3))
segs = []; s = 0
for i in range(1, n + 1):
    if i == n or h2[i] != h2[s]: segs.append((s, i, bool(h2[s]))); s = i
S = pd.DataFrame(segs, columns=['start', 'end', 'high']); S['dur'] = S.end - S.start
print(S[S.high].to_string())
print('volatile durations', np.sort(S[S.high].dur.to_numpy())); print('calm durations', np.sort(S[~S.high].dur.to_numpy()))
# taker rates using the sharper labels
for hh in [False, True]:
    m = (h2 == hh) & mask
    print(f'envW-regime volatile={hh}: secs {m.sum()} taker buy {z["tb"][m].mean():.3f} sell {z["ts"][m].mean():.3f} maker {z["mk"][m].mean():.2f}')
np.save(C / 'regime_envW.npy', h2); S.to_csv(D / 'regimes_envW.csv', index=False)
