"""Calm-only 4 s S increments (ex 5 s around shocks/vol), robust spread in agent episodes vs TWAP vs neither."""
import numpy as np, pandas as pd
S = pd.read_parquet('../s_dynamics/s_path.parquet')
st = S.t_us.to_numpy(); sh = S.s_hat.to_numpy()
V = pd.read_csv('../s_dynamics/vol_segments.csv'); K = pd.read_csv('../flow/shock_events.csv')
print(K.columns.tolist())
e = pd.read_csv('r15_episodes.csv')
tw = pd.read_csv('../novelty/twap_episodes.csv')
H = 4.0
grid = np.arange(0, 6000 - H, H)
def s(t): return sh[np.clip(np.searchsorted(st, t * 1e6, 'right') - 1, 0, len(st) - 1)]
d = s(grid + H) - s(grid)
bad = np.zeros(len(grid), bool)
for r in V.itertuples(): bad |= (grid + H > r.start - 5) & (grid < r.end + 5)
on = K.onset.values
for x in on: bad |= (grid + H > x - 2) & (grid < x + 10)
def cov(iv, c0, c1):
    m = np.zeros(len(grid), bool)
    for r in iv.itertuples(): m |= (grid >= getattr(r, c0)) & (grid + H <= getattr(r, c1))
    return m
ag = cov(e.assign(t0=e.t0 / 1e6, t1=e.t1 / 1e6), 't0', 't1')
twm = cov(tw, 't0', 't1')
for lab, m in [('agent', ag & ~bad), ('TWAP', twm & ~bad & ~ag), ('neither', ~bad & ~ag & ~twm)]:
    x = d[m]
    q = np.quantile(np.abs(x), [.5, .75, .9, .99])
    print(f'{lab:8s} n {m.sum():5d}  |dS_4s| quantiles 50/75/90/99 {q.round(2)}  share >3: {(np.abs(x) > 3).mean():.3f}')
