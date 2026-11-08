"""Hurst (fixed) + robustness of the momentum IOC round trip: latency, shock exclusion, regime, size at the touch."""
import sys
import numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import r4_angles as A

pd.set_option('display.width', 250)
for csv in sys.argv[1:]:
    name = csv.split('/')[-1].replace('_public.csv', '')
    D = A.prep(csv); A.hurst(D, name); st, en = A.shocks(D, name)
    ev = D['ev']; Tus = D['Tus']
    AQ = ev.ask_qty.to_numpy().astype(float); BQ = ev.bid_qty.to_numpy().astype(float)
    at, BID, ASK, dur, turb = D['at'], D['BID'], D['ASK'], D['dur'], D['turb']
    g = np.arange(5, dur - 70, 0.05)
    rows = []
    for (L, X, H) in ((1, 10, 1), (1, 10, 2), (0.5, 5, 1), (0.5, 3, 1), (1, 5, 2), (2, 20, 2)):
        mv = at(g) - at(g - L)
        idx = np.flatnonzero(np.abs(mv) >= X); sel = []; last = -1e9
        for i in idx:
            if g[i] >= last + H: sel.append(i); last = g[i]
        tt = g[np.array(sel)]; sg = np.sign(mv[np.array(sel)])
        near = np.zeros(len(tt), bool)
        for a in st: near |= (tt >= a - 0.5) & (tt <= a + 8)
        reg = turb[np.clip(tt.astype(int) - 1, 0, len(turb) - 1)]       # causal: previous second's flag
        for lat in (0.005, 0.02, 0.05, 0.1, 0.25):
            ent = np.where(sg > 0, at(tt + lat, ASK), at(tt + lat, BID))
            ex = np.where(sg > 0, at(tt + lat + H, BID), at(tt + lat + H, ASK))
            pnl = sg * (ex - ent)
            r = dict(rule=f'L{L} X{X} H{H}', lat_ms=int(lat * 1000), n=len(pnl), pnl=pnl.mean(),
                     t=pnl.mean() / pnl.std() * np.sqrt(len(pnl)))
            if lat == 0.005:
                qe = np.where(sg > 0, at(tt + lat, AQ), at(tt + lat, BQ))
                r |= dict(no_shock=pnl[~near].mean(), n_shock=near.sum(), shock=pnl[near].mean() if near.any() else np.nan,
                          calm_trig=pnl[~reg & ~near].mean(), n_calm=(~reg & ~near).sum(), turb_trig=pnl[reg & ~near].mean(),
                          touch_qty_med=np.median(qe), up=pnl[sg > 0].mean(), down=pnl[sg < 0].mean())
            rows.append(r)
    print(f'\n===== {name} momentum robustness'); print(pd.DataFrame(rows).round(3).to_string(index=False))
print('\n===== hurst'); print(pd.DataFrame(A.RES['hurst']).round(3).to_string())
