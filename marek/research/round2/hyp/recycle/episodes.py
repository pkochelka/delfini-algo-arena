"""Overshoot episodes (|pos| > 100) from a logged backtest: duration, size, entry/exit vs true S."""
import sys, numpy as np, pandas as pd
tr = pd.read_csv(sys.argv[1] + '/trades.csv')
u = tr[(tr.taker_owner == 'user') | (tr.maker_owner == 'user')].copy()
buy = ((u.taker_owner == 'user') & (u.taker_side == 'B')) | ((u.maker_owner == 'user') & (u.taker_side == 'A'))
u['sq'] = np.where(buy, u.quantity, -u.quantity); u['t'] = u.timestamp_us * 1e-6; u['pos'] = u.sq.cumsum()
S = pd.read_parquet('marek/research/round2/s_dynamics/s_path.parquet', columns=['t_us', 's_hat'])
u['S'] = np.interp(u.timestamp_us, S.t_us, S.s_hat)
u['S1'] = np.interp(u.timestamp_us + 1e6, S.t_us, S.s_hat)
u['edge'] = (u.S - u.price) * u.sq
u['edge1'] = (u.S1 - u.price) * u.sq
over = (u.pos.abs() > 100).values
eps = []; i = 0; n = len(u)
while i < n:
    if over[i]:
        j = i
        while j + 1 < n and over[j + 1]: j += 1
        a = max(i - 3, 0); seg = u.iloc[i:j + 2]
        eps.append(dict(t=u.t.iloc[i], dur=(u.t.iloc[min(j + 1, n - 1)] - u.t.iloc[i]), peak=u.pos.iloc[i:j + 1].abs().max(),
                        nfill=j + 2 - i, edge=seg.edge.sum(), edge1=seg.edge1.sum()))
        i = j + 1
    i += 1
e = pd.DataFrame(eps)
pd.set_option('display.width', 200)
print(len(e), 'episodes; total dur', e.dur.sum(), 'edge', e.edge.sum(), 'edge1', e.edge1.sum())
print(e.describe().round(3))
print(e.sort_values('dur', ascending=False).head(15).round(2))
