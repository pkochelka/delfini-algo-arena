import sys, numpy as np, pandas as pd
f = pd.read_csv('marek/research/round2/hyp/recycle/capture_first.csv')
d = pd.read_csv('marek/research/round2/hyp/recycle/stale_events.csv')
tr = pd.read_csv(sys.argv[1] + '/trades.csv')
u = tr[(tr.taker_owner == 'user') | (tr.maker_owner == 'user')].copy()
buy = ((u.taker_owner == 'user') & (u.taker_side == 'B')) | ((u.maker_owner == 'user') & (u.taker_side == 'A'))
u['sq'] = np.where(buy, u.quantity, -u.quantity); u['t'] = u.timestamp_us * 1e-6; u['pos'] = u.sq.cumsum()
tt, pp = u.t.values, u.pos.values
def pos(t):
    i = np.searchsorted(tt, t); return pp[i-1] if i else 0
f['sg'] = np.sign(f.J)
for lag in [0.01, 0.05, 0.2, 1, 5, 30]:
    f[f'p{lag}'] = [pos(t - lag) * s for t, s in zip(f.t, f.sg)]
pd.set_option('display.width', 250)
print(f[f.room <= 20][['t', 'J', 'V', 'p0', 'p0.01', 'p0.05', 'p0.2', 'p1', 'p5', 'p30', 'shock']].head(40).round(2))
