"""Ceiling angle: queue position of r3_best's passive fills. For each market taker that filled one of our resting
orders: market lots filled at our price ahead of us vs behind us, and taker lots at our price that we lost because our
order was not there."""
import sys, pathlib
import numpy as np, pandas as pd
H = pathlib.Path(__file__).parent
sys.path.insert(0, str(H.parents[2] / 'strategy'))
import eval3
on, reg = eval3.load_ctx('data/round3.csv')
tr = pd.read_csv(sys.argv[1])
tr = tr[tr.taker_owner == 'market']
tr['seq'] = np.arange(len(tr))
us = tr[tr.maker_owner == 'user']
ids = set(us.taker_order_id)
x = tr[tr.taker_order_id.isin(ids)]
res = {r: [0, 0, 0] for r in ['calm', 'volatile', 'shock']}
for tid, g in x.groupby('taker_order_id', sort=False):
    u = g[g.maker_owner == 'user']
    px = u.price.iloc[0]; s0 = u.seq.min()
    same = g[(g.price == px) & (g.maker_owner == 'market')]
    r = eval3.label(g.timestamp_us.iloc[0] * 1e-6, on, reg)
    res[r][0] += u.quantity.sum(); res[r][1] += same[same.seq < s0].quantity.sum(); res[r][2] += same[same.seq > s0].quantity.sum()
print('regime: our lots, market lots at our price filled AHEAD of us, market lots at our price filled after us')
for r, v in res.items():
    print(r, v)
