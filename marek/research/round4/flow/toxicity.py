"""Maker-side cost of the side law: per filled lot, taker edge vs S at the trade and S markouts at +1/5/20 s, by
xa = sg*(S - mid) (alignment) and regime. Edge = sg*(S_h - avg fill price) (positive = taker wins, maker loses)."""
import numpy as np, pandas as pd
from pathlib import Path
C = Path(__file__).parent / 'cache'
tk = pd.read_parquet(C / 'takers_s.parquet'); z = np.load(C / 's_series.npz'); tr = pd.read_pickle(C / 'trades.pkl')
o = pd.read_pickle(C / 'orders.pkl'); r = pd.read_pickle(C / 'rows.pkl')
S_at = lambda t: z['s'][np.maximum(np.searchsorted(z['t'], t) - 1, 0)]
# avg fill price per taker: trades keyed by taker_row (row index in csv)
rowidx = np.flatnonzero((r.op == 0).to_numpy() & r.ioc.to_numpy())
g = tr.assign(pq=tr.price * tr.qty).groupby('taker_row')[['pq', 'qty']].sum()
avgp = (g.pq / g.qty).reindex(rowidx).to_numpy(); q = g.qty.reindex(rowidx).fillna(0).to_numpy()
t = tk.t.to_numpy(); s = tk.sg.to_numpy(); xa = s * (tk.s_hat.to_numpy() - tk.mid.to_numpy()); reg = tk.reg.to_numpy()
ok = (q > 0) & ((tk.shock_dt < -1) | (tk.shock_dt > 30)).to_numpy() & np.isfinite(xa) & (t < 5970)
for R in [0, 1]:
    print(f'== regime {R}  (lot-weighted, S-ticks per lot; nonTWAP)')
    for a, b in [(-99, -1), (-1, 0), (0, 1), (1, 2), (2, 99)]:
        m = ok & (reg == R) & (xa >= a) & (xa < b) & ~tk.tw.to_numpy(); w = q[m]
        row = [f'xa [{a},{b}) lots share {w.sum() / q[ok & (reg == R)].sum():.3f}']
        for h in [0, 1, 5, 20]:
            Sh = tk.s_hat.to_numpy()[m] if h == 0 else S_at(t[m] + h)
            row.append(f'h{h}: {np.sum(w * s[m] * (Sh - avgp[m])) / w.sum():+.2f}')
        print('   ' + '  '.join(row))
    m = ok & (reg == R) & tk.tw.to_numpy(); w = q[m]
    print('   TWAP:', '  '.join(f'h{h}: {np.sum(w * s[m] * ((tk.s_hat.to_numpy()[m] if h == 0 else S_at(t[m] + h)) - avgp[m])) / w.sum():+.2f}' for h in [0, 1, 5, 20]))
