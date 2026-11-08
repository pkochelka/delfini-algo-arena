"""Inventory PnL (sum inv * dS on exact S) around shocks for a run: drought, across the jump, and after it.
usage: uv run --with pyarrow python shock_inv.py RUN_LOGDIR
"""
import sys, numpy as np, pandas as pd
rd = sys.argv[1]
S = pd.read_parquet('marek/research/round3/s_dynamics/s_path.parquet', columns=['t_us', 's_hat'])
sh = pd.read_csv('marek/research/round3/s_dynamics/shocks.csv')
st = S.t_us.values; shv = S.s_hat.values
cols = ['ts', 'price', 'tside', 'qq', 'tid', 'mid', 'town', 'mown']
rt = pd.read_csv(f'{rd}/trades.csv', names=cols, header=0)
u = rt[(rt.town == 'user') | (rt.mown == 'user')]
sgn = np.where(((u.town == 'user') & (u.tside == 'B')) | ((u.mown == 'user') & (u.tside == 'A')), 1, -1) * u.qq.values
ut = u.ts.values; cum = np.cumsum(sgn)
def inv_at(t):
    i = np.searchsorted(ut, t, side='right') - 1
    return np.where(i >= 0, cum[np.clip(i, 0, None)], 0)
def s_at(t):
    return shv[np.clip(np.searchsorted(st, t, side='right') - 1, 0, None)]
def invpnl(a, b, step=10_000):
    g = np.arange(a, b, step); s = s_at(g); i = inv_at(g)
    return float((i[:-1] * np.diff(s)).sum())
rows = []
for _, r in sh.iterrows():
    on = int(r.onset * 1e6); tj = int(r.t_jump * 1e6)
    rows.append(dict(J=round(r.jump, 1), inv_jump=int(inv_at(tj - 2000)), drought=invpnl(on - 1_000_000, tj - 2000),
                     jump=float(inv_at(tj - 2000) * (s_at(tj + 30_000) - s_at(tj - 2000))),
                     post0_1=invpnl(tj + 30_000, tj + 1_000_000), post1_20=invpnl(tj + 1_000_000, tj + 17_000_000)))
d = pd.DataFrame(rows).round(0)
print(d.to_string()); print(d.sum())
