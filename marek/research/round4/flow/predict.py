"""What predicts the next taker side and the next mid move, in real time? 0.1 s grid, ex-shock, by regime.
Features: S - mid (exact S, and causal s_filt), EWMA taker imbalance (tau 0.25/1 s, non-TWAP), last unfilled IOC, past mid
moves. Targets: next-taker sign (per taker), mid change over next 1 / 5 / 20 s (grid)."""
import numpy as np, pandas as pd
from pathlib import Path
D = Path(__file__).parent; C = D / 'cache'
tk = pd.read_parquet(C / 'takers_s.parquet'); z = np.load(C / 's_series.npz'); r = pd.read_pickle(C / 'rows.pkl')
ev = pd.read_csv(D / 'shock_events.csv'); vs = pd.read_csv(D.parent / 's_dynamics/vol_segments.csv')
rt = r.t.to_numpy(); mid = ((r.bid_after + r.ask_after) / 2).ffill().to_numpy()
mid_at = lambda t: mid[np.maximum(np.searchsorted(rt, t) - 1, 0)]
S_at = lambda t, key='s': z[key][np.maximum(np.searchsorted(z['t'], t) - 1, 0)]
g = np.arange(30, 5980, 0.1); reg = np.zeros(len(g), int)
for a, b in zip(vs.start, vs.end): reg[(g >= a) & (g < b)] = 1
ok = np.ones(len(g), bool)
for o in ev.onset: ok &= ~((g > o - 1) & (g < o + 25))
m0 = mid_at(g); F = pd.DataFrame({'S_mid': S_at(g) - m0, 'Sf_mid': S_at(g, 'sf') - m0})
t = tk.t.to_numpy(); s = tk.sg.to_numpy().astype(float); nt = ~tk.tw.to_numpy()
for tau in [0.25, 1.0]:
    e = np.zeros(len(t)); acc = 0.; 
    for i in range(1, len(t)): acc = acc * np.exp(-(t[i] - t[i - 1]) / tau) + (s[i - 1] if nt[i - 1] else 0); e[i] = acc
    j = np.searchsorted(t, g) - 1; F[f'imb{tau}'] = (e[np.maximum(j, 0)] + np.where(nt[np.maximum(j, 0)], s[np.maximum(j, 0)], 0)) * np.exp(-(g - t[np.maximum(j, 0)]) / tau)
uf = (tk.filled.to_numpy() == 0); tu = t[uf]; su = s[uf]; j = np.searchsorted(tu, g) - 1
F['unfilled1s'] = np.where((j >= 0) & (g - tu[np.maximum(j, 0)] < 1.0), -su[np.maximum(j, 0)], 0)  # +1 = last unfilled was a sell (=> buys next)
for h in [0.5, 2]: F[f'dmid{h}'] = m0 - mid_at(g - h)
Y = {f'fwd{h}': mid_at(g + h) - m0 for h in [1, 5, 20]}
for R in [0, 1]:
    q = ok & (reg == R) & np.isfinite(F.to_numpy()).all(1)
    print(f'==== regime {R}  n grid {q.sum()}')
    for yn, y in Y.items():
        print(f'  {yn}: corr', {c: round(np.corrcoef(F.loc[q, c], y[q])[0, 1], 3) for c in F.columns}, end='')
        for cols in [['imb0.25', 'imb1.0', 'dmid0.5', 'dmid2', 'unfilled1s'], ['Sf_mid'], ['Sf_mid', 'imb0.25', 'imb1.0', 'dmid0.5', 'dmid2', 'unfilled1s']]:
            A = np.c_[np.ones(q.sum()), F.loc[q, cols].to_numpy()]; b = np.linalg.lstsq(A, y[q], rcond=None)[0]
            print(f' | R2[{"+".join(cols)}]={1 - np.var(y[q] - A @ b) / np.var(y[q]):.3f}', end='')
        print()
# next-taker sign from real-time features (per taker, non-TWAP)
tt = t[nt]; ss = s[nt]; G = F.copy(); idx = np.searchsorted(g, tt - 1e-9) - 1  # grid point just before taker (<=0.1 s stale)
for R in [0, 1]:
    rg = np.zeros(len(tt), int)
    for a, b in zip(vs.start, vs.end): rg[(tt >= a) & (tt < b)] = 1
    q = (idx >= 0) & (rg == R) & ok[np.maximum(idx, 0)]
    print(f'reg {R} next-sign corr (feature at grid <=0.1 s before):', {c: round(np.corrcoef(F[c].to_numpy()[idx[q]], ss[q])[0, 1], 3) for c in F.columns})
