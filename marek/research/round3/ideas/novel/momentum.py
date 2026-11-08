"""Short-horizon S momentum (exact S, 100 ms grid, shock windows masked): corr/beta of future move on past move,
and the frictionless PnL of holding 100*sign(past move) (upper bound for a momentum position), per regime.
usage: uv run --with pyarrow python momentum.py
"""
import csv, numpy as np, pandas as pd
S = pd.read_parquet('marek/research/round3/s_dynamics/s_path.parquet', columns=['t_us', 's_hat', 'vol'])
st = S.t_us.values; sh = S.s_hat.values; vol = S.vol.values
grid = np.arange(0, st[-1], 100_000)
ix = np.clip(np.searchsorted(st, grid, side='right') - 1, 0, None)
Sg = sh[ix]; vg = vol[ix]
on = np.array([float(r['onset']) for r in csv.DictReader(open('marek/research/round3/flow/shock_events.csv'))]) * 1e6
bad = np.zeros(len(grid), bool)
for o in on: bad |= (grid >= o - 1e6) & (grid < o + 25e6)
n = len(grid); idx = np.arange(n)
for K in (2, 5, 10, 20, 50):
    past = Sg - np.roll(Sg, K)
    out = []
    for H in (2, 5, 10, 20, 50, 100):
        fut = np.roll(Sg, -H) - Sg
        m = ~bad & (idx > K) & (idx < n - H)
        for nm, mm in (('calm', m & ~vg), ('vol', m & vg)):
            r = np.corrcoef(past[mm], fut[mm])[0, 1]
            out.append(f'{nm} H{H/10:g}s r={r:+.3f}')
    print(f'K={K/10:g}s', ' | '.join(out))
# frictionless holder: pos = 100*sign(past K), earn pos * dS over next step (rebalance every 100 ms)
dS = np.roll(Sg, -1) - Sg
for K in (2, 5, 10, 20):
    past = Sg - np.roll(Sg, K)
    m = ~bad & (idx > K) & (idx < n - 1)
    for nm, mm in (('calm', m & ~vg), ('vol', m & vg)):
        print(f'hold 100*sign(past {K/10:g}s) {nm}: {(100*np.sign(past)*dS)[mm].sum():.0f}  turnover lots {np.abs(np.diff(100*np.sign(past[mm]))).sum():.0f}')
