"""Measure 'zombie' liquidity: market orders our IOCs hit that, in the strategy-free replay, were already filled.
usage: uv run --with pyarrow python zombie_fills.py NULL_LOGDIR RUN_LOGDIR
A zombie lot = a lot of maker order m we take at time t while the null replay has m fully consumed before t
(the taker that consumed it hit our resting order instead, so m outlived its tape life: fully-filled orders have no cancel row).
"""
import sys, numpy as np, pandas as pd
nd, rd = sys.argv[1], sys.argv[2]
cols = ['ts', 'price', 'tside', 'q', 'tid', 'mid', 'town', 'mown']
nt = pd.read_csv(f'{nd}/trades.csv', names=cols, header=0, dtype={'tid': str, 'mid': str})
rt = pd.read_csv(f'{rd}/trades.csv', names=cols, header=0, dtype={'tid': str, 'mid': str})
S = pd.read_parquet('marek/research/round3/s_dynamics/s_path.parquet', columns=['t_us', 's_hat', 'vol', 'shock'])
st = S.t_us.values; sh = S.s_hat.values
def s_at(t):
    i = np.searchsorted(st, t, side='right') - 1
    return sh[np.clip(i, 0, len(sh) - 1)]
# null: cumulative fill per maker order over time -> time it was exhausted (if ever)
tape = pd.read_csv('data/round3.csv', usecols=['Time', 'Operation', 'OrderId', 'Volume'], dtype={'OrderId': str})
ins = tape[tape.Operation == 'Insert'].set_index('OrderId').Volume
canc = set(tape.loc[tape.Operation == 'Cancel', 'OrderId'])
nt['cum'] = nt.groupby('mid').q.cumsum()
nt['vol'] = nt.mid.map(ins)
done = nt[nt.cum >= nt.vol].groupby('mid').ts.first()   # time order exhausted in null replay
print('null: maker orders fully filled', len(done), ' of which have a cancel row:', sum(m in canc for m in done.index))
u = rt[(rt.town == 'user') | (rt.mown == 'user')].copy()
u['S'] = s_at(u.ts.values)
u['sign'] = np.where(((u.town == 'user') & (u.tside == 'B')) | ((u.mown == 'user') & (u.tside == 'A')), 1, -1)  # +1 we buy
u['edge'] = u.sign * (u.S - u.price) * u.q
take = u[u.town == 'user'].copy()
take['exh'] = take.mid.map(done)
take['zombie'] = take.exh.notna() & (take.exh < take.ts)
mk = u[u.mown == 'user']
print(f'user trades {len(u)}  volume {u.q.sum()}  edge vs S_hat {u.edge.sum():.0f}')
print(f'  as taker: vol {take.q.sum()} edge {take.edge.sum():.0f};  as maker: vol {mk.q.sum()} edge {mk.edge.sum():.0f}')
z = take[take.zombie]
print(f'  taker fills on zombies: n {len(z)} vol {z.q.sum()} edge {z.edge.sum():.0f}  (age since null exhaustion median {np.median(z.ts - z.exh)/1e3:.0f} ms)')
# zombie mass in book over time: our passive volume - our take volume is roughly the excess liquidity.
