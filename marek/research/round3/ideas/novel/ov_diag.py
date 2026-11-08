"""Our passive fills by (our price - taker limit), signed so that 0 = at the limit, -1 = one tick inside reach.
usage: uv run --with pyarrow python ov_diag.py RUN_LOGDIR
"""
import sys, numpy as np, pandas as pd
rd = sys.argv[1]
S = pd.read_parquet('marek/research/round3/s_dynamics/s_path.parquet', columns=['t_us', 's_hat', 'vol', 'shock'])
st = S.t_us.values; sh = S.s_hat.values; vol = S.vol.values; shk = S.shock.values
tape = pd.read_csv('data/round3.csv', usecols=['Operation', 'OrderId', 'Price', 'Volume'])
ins = tape[tape.Operation == 'Insert']
lim = dict(zip(ins.OrderId.values, ins.Price.values)); qv = dict(zip(ins.OrderId.values, ins.Volume.values))
cols = ['ts', 'price', 'tside', 'qq', 'tid', 'mid', 'town', 'mown']
rt = pd.read_csv(f'{rd}/trades.csv', names=cols, header=0, dtype={'tid': str, 'mid': str})
p = rt[rt.mown == 'user'].copy()
i = np.clip(np.searchsorted(st, p.ts.values, side='right') - 1, 0, None)
REG = sys.argv[2] if len(sys.argv) > 2 else 'vol'
p = p[(vol[i] if REG == 'vol' else ~vol[i]) & ~shk[i]]
sg = np.where(p.tside.values == 'B', 1, -1)    # taker side
L = np.array([lim.get(int(x), np.nan) for x in p.tid.values])
p['o'] = sg * (L - p.price.values) * -1         # 0 at limit; -1 = our price one tick inside the taker's reach
p['tq'] = [qv.get(int(x), 0) for x in p.tid.values]
def s_at(x):
    return sh[np.clip(np.searchsorted(st, x, side='right') - 1, 0, None)]
p['s0'] = sh[np.clip(np.searchsorted(st, p.ts.values, side='left') - 1, 0, None)]
p['d'] = sg * (p.price.values - p.s0)
for h in (1, 30):
    p[f'e{h}'] = sg * (p.price.values - s_at(p.ts.values + h * 1e6)) * p.qq
g = p.groupby('o').agg(n=('qq', 'size'), lots=('qq', 'sum'), d=('d', 'mean'), tq=('tq', 'median'), e1=('e1', 'sum'), e30=('e30', 'sum'))
print(g.round(2).to_string())
