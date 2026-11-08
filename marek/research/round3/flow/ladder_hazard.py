"""Ladder-start hazard per side on a 10 ms grid vs state: regime, signed S move over last 0.25 s (+ = away from the
side, i.e. S up for bids), gap between S and same-side touch, same-side touch moved recently, time since last ladder."""
import numpy as np, pandas as pd
from pathlib import Path
D = Path(__file__).parent; C = D / 'cache'
r = pd.read_pickle(C / 'rows.pkl'); ins = r[r.op == 0].reset_index(drop=True)
S = pd.read_parquet(D.parent / 's_dynamics/s_path.parquet'); L = pd.read_pickle(C / 'ladders.pkl')
ti = ins.t.to_numpy(); s = S.s_hat.to_numpy(); vol = S.vol.to_numpy(); shock = S.shock.to_numpy()
W = 0.01; g = np.arange(0, 6000, W) + W  # grid points (state measured just before)
k = np.clip(np.searchsorted(ti, g) - 1, 0, None)
Sg = s[k]; Vg = vol[k]; Hg = shock[k]
k25 = np.clip(np.searchsorted(ti, g - 0.25) - 1, 0, None); dS = Sg - s[k25]
kr = np.clip(np.searchsorted(r.t.to_numpy(), g) - 1, 0, None)
bid = r.bid_after.to_numpy()[kr]; ask = r.ask_after.to_numpy()[kr]
out = []
for side in [0, 1]:
    sgn = 1 if side == 0 else -1  # away from bid side = S up
    lt = np.sort(L.t0[L.side == side].to_numpy())
    cnt = np.bincount((lt / W).astype(int), minlength=len(g))[:len(g)]  # starts in (g-W, g] -> use next bin
    nxt = np.r_[cnt[1:], 0]
    away = sgn * dS
    gapS = np.where(side == 0, Sg - bid, ask - Sg)  # distance from S to own touch
    lastL = np.r_[-np.inf, lt]; tl = g - lastL[np.searchsorted(lt, g)]
    out.append(pd.DataFrame({'n': nxt, 'vol': Vg, 'shock': Hg, 'away': away, 'gapS': gapS, 'tl': tl}))
X = pd.concat(out); X = X[~X.shock]
pd.set_option('display.width', 200)
for v in [False, True]:
    Y = X[X.vol == v]; print(f'==== vol={v}: base hazard per side {Y.n.sum() / len(Y) / W:.3f}/s')
    print(' by S move over last 0.25 s (+ away from side):'); print((Y.groupby(pd.cut(Y.away, [-50, -3, -1.5, -0.5, -0.1, 0.1, 0.5, 1.5, 3, 50])).n.agg(['mean', 'size']).assign(rate=lambda d: d['mean'] / W)).round(3).to_string())
    print(' by S - own touch (ticks):'); print((Y.groupby(pd.cut(Y.gapS, [-50, 0, 0.5, 1, 1.5, 2, 3, 5, 50])).n.agg(['mean', 'size']).assign(rate=lambda d: d['mean'] / W)).round(3).to_string())
    print(' by time since last same-side ladder:'); print((Y.groupby(pd.cut(Y.tl, [0, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 1e9])).n.agg(['mean', 'size']).assign(rate=lambda d: d['mean'] / W)).round(3).to_string())
X.to_pickle(C / 'ladder_hazard_grid.pkl')
