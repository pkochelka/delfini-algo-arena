"""Aligned vs against taker rates as a function of S-mid (0.05 s grid, non-TWAP, ex-shock, regime interiors >= 10 s from
switches, outside TWAP windows). Side-law model: total constant, against falls; additive arbitrage class: against flat."""
import numpy as np, pandas as pd
from pathlib import Path
D = Path(__file__).parent; C = D / 'cache'
tk = pd.read_parquet(C / 'takers_s.parquet'); R = pd.read_csv(D / 'regimes.csv'); ev = pd.read_csv(D / 'shock_events.csv')
tw = pd.read_csv(D.parent / 'novelty/twap_episodes.csv')
dt = 0.05; g = np.arange(0, 6000, dt); reg = np.zeros(len(g), int); ok = np.ones(len(g), bool)
for _, r in R.iterrows():
    reg[(g >= r.start) & (g < r.end)] = int(r.regime == 'volatile'); ok &= ~((g > r.start - 10) & (g < r.start + 10))
for _, e in ev.iterrows(): ok &= ~((g > e.t_jump - 4) & (g < e.t_jump + 8))
for _, w in tw.iterrows(): ok &= ~((g > w.t0 - 1) & (g < w.t1 + 1))
z = np.load(C / 's_series.npz'); r = pd.read_pickle(C / 'rows.pkl'); rt = r.t.to_numpy(); mid = ((r.bid_after + r.ask_after) / 2).ffill().to_numpy()
xg = z['s'][np.maximum(np.searchsorted(z['t'], g) - 1, 0)] - mid[np.maximum(np.searchsorted(rt, g) - 1, 0)]
nt = ~tk.tw.to_numpy(); t = tk.t.to_numpy(); gi = np.minimum((t / dt).astype(int), len(g) - 1); s = tk.sg.to_numpy()
cb = np.bincount(gi[nt & (s > 0)], minlength=len(g)); cs = np.bincount(gi[nt & (s < 0)], minlength=len(g))
al = np.where(xg >= 0, cb, cs); ag = np.where(xg >= 0, cs, cb)
for Rg in [0, 1]:
    q = ok & (reg == Rg) & np.isfinite(xg); print(f'regime {Rg}: grid {q.sum()}')
    for a, b in [(0, .25), (.25, .5), (.5, 1), (1, 1.5), (1.5, 2), (2, 3), (3, 5), (5, 99)]:
        m = q & (np.abs(xg) >= a) & (np.abs(xg) < b)
        if m.sum() > 100: print(f'   |x| {a}-{b}: n {m.sum():6d}  aligned {al[m].mean() / dt:6.2f}/s  against {ag[m].mean() / dt:6.2f}/s  total {(al[m] + ag[m]).mean() / dt:6.2f}  aligned share {al[m].sum() / max((al[m] + ag[m]).sum(), 1):.3f}')
