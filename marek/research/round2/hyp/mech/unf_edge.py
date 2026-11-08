import pandas as pd, numpy as np
tk = pd.read_pickle('marek/research/round2/flow/cache/takers.pkl')
tr = pd.read_pickle('marek/research/round2/flow/cache/trades.pkl')
sp = pd.read_parquet('marek/research/round2/s_dynamics/s_path.parquet')
f = tr.groupby('taker_row').qty.sum()
tk['filled'] = f.reindex(tk.index).fillna(0).values
tk['unf'] = tk.vol - tk.filled
ts = sp.t_us.values; S = sp.s_hat.values
tus = np.round(tk.t.values * 1e6).astype(np.int64)
i = np.searchsorted(ts, tus)  # index of this taker's insert
def s_at(dt):
    j = np.clip(np.searchsorted(ts, tus + dt), 0, len(ts) - 1)
    return S[j]
sgn = np.where(tk.side == 0, 1, -1)  # +1 buy taker
tk['S0'] = S[np.clip(i - 1, 0, len(S) - 1)]
tk['S1'] = S[np.clip(i + 1, 0, len(S) - 1)]
for dt, nm in [(100_000, 'S100ms'), (1_000_000, 'S1s'), (10_000_000, 'S10s')]:
    tk[nm] = s_at(dt)
# we sell to a buy taker at its limit px: edge = sgn*(px - S)
for c in ['S0', 'S1', 'S100ms', 'S1s', 'S10s']:
    tk['e_' + c] = sgn * (tk.px - tk[c])
tk['lim_off'] = np.where(tk.side == 0, tk.px - tk.ask_before, tk.bid_before - tk.px)
u = tk[tk.unf > 0].copy()
w = u.unf
print('total unfilled', w.sum())
for c in ['S0', 'S1', 'S100ms', 'S1s', 'S10s']:
    print(c, 'vol-weighted edge of unfilled remainder at limit px', (u['e_' + c] * w).sum() / w.sum(), 'tick*lots', (u['e_' + c] * w).sum())
g = u.groupby([u.hi, u.shock, u.lim_off.clip(-2, 3)])
print(g.apply(lambda d: pd.Series({'n': len(d), 'unf': d.unf.sum(), 'e0': (d.e_S0 * d.unf).sum() / d.unf.sum(),
      'e1': (d.e_S1 * d.unf).sum() / d.unf.sum(), 'e1s': (d.e_S1s * d.unf).sum() / d.unf.sum(), 'e10s': (d.e_S10s * d.unf).sum() / d.unf.sum()})).round(2))
# for comparison: all filled taker volume edge at their fill (vwap approx by px? use touch)
