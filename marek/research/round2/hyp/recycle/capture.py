"""fvs_mm capture per jump event vs the stale edge available (stale_events.csv). arg: backtest log dir."""
import sys, numpy as np, pandas as pd
d = pd.read_csv('marek/research/round2/hyp/recycle/stale_events.csv')
tr = pd.read_csv(sys.argv[1] + '/trades.csv')
u = tr[(tr.taker_owner == 'user') | (tr.maker_owner == 'user')].copy()
# our signed qty: user buys if (taker user and taker_side B) or (maker user and taker_side A)
buy = ((u.taker_owner == 'user') & (u.taker_side == 'B')) | ((u.maker_owner == 'user') & (u.taker_side == 'A'))
u['sq'] = np.where(buy, u.quantity, -u.quantity)
u['t'] = u.timestamp_us * 1e-6
u['pos'] = u.sq.cumsum()
tt = u.t.values; pp = u.pos.values
rows = []
for e in d.itertuples():
    i0 = np.searchsorted(tt, e.t - 0.0)
    p0 = pp[i0 - 1] if i0 else 0
    w = u[(u.t >= e.t) & (u.t < e.t + 0.3)]
    sgn = 1 if e.up else -1
    good = w[np.sign(w.sq) == sgn]
    q = good.sq.abs().sum()
    edge = ((e.S - good.price) * good.sq).sum()
    rows.append(dict(t=e.t, J=e.J, V=e.V, E100=e.E100, E200=e.E200, p0=p0 * sgn, room=100 - p0 * sgn, q=q, edge=edge, shock=e.shock))
r = pd.DataFrame(rows)
pd.set_option('display.width', 200)
print(r.describe().round(1))
print('captured edge', r.edge.sum(), 'E100', r.E100.sum(), 'E200', r.E200.sum())
r['Eroom'] = 0.0
print('room==0 events', (r.room <= 0).sum(), 'room>=150', (r.room >= 150).sum())
print(r.groupby(pd.cut(r.room, [-1, 20, 80, 120, 180, 201])).agg(n=('q', 'size'), q=('q', 'mean'), edge=('edge', 'sum'), V=('V', 'median')))
print(r.groupby('shock').agg(edge=('edge', 'sum'), E100=('E100', 'sum'), E200=('E200','sum')))
gap = np.r_[1e9, np.diff(r.t.values)]
f = r[gap > 1.0].copy()
print('--- first-in-cluster events', len(f))
print(f.groupby(pd.cut(f.room, [-1, 20, 80, 120, 180, 201])).agg(n=('q', 'size'), q=('q', 'mean'), edge=('edge', 'sum'), E100=('E100','sum'), E200=('E200','sum'), V=('V', 'median')))
print('p0 in jump dir mean', f.p0.mean(), ' sign corr', np.mean(np.sign(f.p0)))
f.to_csv('marek/research/round2/hyp/recycle/capture_first.csv', index=False)
