"""Maker-side markout of tape trades by the trade's distance from S (before the taker), per regime.
A resting order at distance d earns roughly this per lot when filled (the queue ahead aside).
usage: uv run --with pyarrow python depth_markout.py NULL_LOGDIR
"""
import sys, csv, numpy as np, pandas as pd
nd = sys.argv[1]
S = pd.read_parquet('marek/research/round3/s_dynamics/s_path.parquet', columns=['t_us', 's_hat', 'vol'])
st = S.t_us.values; sh = S.s_hat.values; vol = S.vol.values
cols = ['ts', 'price', 'tside', 'q', 'tid', 'mid', 'town', 'mown']
t = pd.read_csv(f'{nd}/trades.csv', names=cols, header=0)
t = t[t.town == 'market']
i = np.searchsorted(st, t.ts.values, side='left') - 1          # S just before the taker's insert row
s0 = sh[np.clip(i, 0, None)]; v = vol[np.clip(i, 0, None)]
on = np.array([float(r['onset']) for r in csv.DictReader(open('marek/research/round3/flow/shock_events.csv'))]) * 1e6
bad = np.zeros(len(t), bool)
for o in on: bad |= (t.ts.values >= o - 1e6) & (t.ts.values < o + 20e6)
sg = np.where(t.tside.values == 'B', 1, -1)      # taker side; maker is the other
d = sg * (t.price.values - s0)                   # how far through S the maker's price was (positive = maker priced away from S)
def s_at(x):
    return sh[np.clip(np.searchsorted(st, x, side='right') - 1, 0, None)]
q = t.q.values
df = pd.DataFrame(dict(v=v, d=np.clip(np.round(d), -3, 8), q=q, bad=bad))
for h in (1, 10, 30):
    df[f'e{h}'] = sg * (t.price.values - s_at(t.ts.values + h * 1e6)) * q
# taker total filled per taker (to see who reaches deep)
df = df[~df.bad]
g = df.groupby(['v', 'd']).agg(lots=('q', 'sum'), e1=('e1', 'sum'), e10=('e10', 'sum'), e30=('e30', 'sum'))
for c in ('e1', 'e10', 'e30'): g[c] = (g[c] / g.lots).round(2)
print(g.to_string())
