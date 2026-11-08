"""Calm: passive fill rate and edge of a run, by |inventory| bucket at fill time (time-weighted denominators).
usage: uv run --with pyarrow python calm_inv.py RUN_LOGDIR
"""
import sys, csv, bisect, numpy as np, pandas as pd
rd = sys.argv[1]
S = pd.read_parquet('marek/research/round3/s_dynamics/s_path.parquet', columns=['t_us', 's_hat', 'vol'])
st = S.t_us.values; shv = S.s_hat.values; vol = S.vol.values
s_at = lambda t: shv[np.clip(np.searchsorted(st, t, side='right') - 1, 0, None)]
v_at = lambda t: vol[np.clip(np.searchsorted(st, t, side='right') - 1, 0, None)]
on = np.array([float(r['onset']) for r in csv.DictReader(open('marek/research/round3/flow/shock_events.csv'))]) * 1e6
cols = ['ts', 'price', 'tside', 'qq', 'tid', 'mid', 'town', 'mown']
rt = pd.read_csv(f'{rd}/trades.csv', names=cols, header=0)
u = rt[(rt.town == 'user') | (rt.mown == 'user')].copy()
u['sg'] = np.where(((u.town == 'user') & (u.tside == 'B')) | ((u.mown == 'user') & (u.tside == 'A')), 1, -1)
u['cum'] = np.cumsum(u.sg * u.qq); u['pre'] = u.cum - u.sg * u.qq
grid = np.arange(0, st[-1], 100_000)
ui = np.searchsorted(u.ts.values, grid, side='right') - 1
invg = np.where(ui >= 0, u.cum.values[np.clip(ui, 0, None)], 0)
shock = np.zeros(len(grid), bool)
for o in on: shock |= (grid >= o - 1e6) & (grid < o + 20e6)
calm = ~v_at(grid) & ~shock
def shk(t):
    m = np.zeros(len(t), bool)
    for o in on: m |= (t >= o - 1e6) & (t < o + 20e6)
    return m
p = u[(u.mown == 'user')].copy()
p = p[~v_at(p.ts.values) & ~shk(p.ts.values)]
p['e30'] = p.sg * (s_at(p.ts.values + 30e6) - p.price) * p.qq
p['e1'] = p.sg * (s_at(p.ts.values + 1e6) - p.price) * p.qq
p['reduce'] = np.sign(p.pre) == -p.sg
bins = [0, 20, 50, 80, 99, 1000]
p['b'] = pd.cut(p.pre.abs(), bins, right=False); tb = pd.cut(np.abs(invg[calm]), bins, right=False)
secs = pd.Series(tb).value_counts().sort_index() * 0.1
g = p.groupby(['b', 'reduce'], observed=True).agg(vol=('qq', 'sum'), e1=('e1', 'sum'), e30=('e30', 'sum')).unstack()
g['secs'] = secs.values
print(g.round(0).to_string())
print('calm seconds', calm.sum() * 0.1, 'mean |inv|', np.abs(invg[calm]).mean())
