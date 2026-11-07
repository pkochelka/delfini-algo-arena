"""Basic structure + time-varying rates for makers / takers per side."""
import numpy as np, pandas as pd
from pathlib import Path
C = Path(__file__).parent / 'cache'
r = pd.read_pickle(C / 'rows.pkl')
ins = r[r.op == 0].copy()
print('rows', len(r), 'inserts', len(ins))
oid = ins.oid.to_numpy().astype(np.int64)
d = np.diff(oid); print('oid first', oid[0], 'diff values', np.unique(d, return_counts=True))
t = r.t.to_numpy(); print('dup timestamps', (np.diff(t) == 0).sum(), 'non-monotone', (np.diff(t) < 0).sum())
# timestamp microsecond digit pattern
us = np.rint(t * 1e6).astype(np.int64); print('t us mod 10 distribution', np.bincount(us % 10))
T = 6000
mk = ins[~ins.ioc]; tk = ins[ins.ioc]
streams = {'maker_bid': mk[mk.side == 0].t, 'maker_ask': mk[mk.side == 1].t,
           'taker_buy': tk[tk.side == 0].t, 'taker_sell': tk[tk.side == 1].t}
for w in [1, 10, 60, 300]:
    print(f'--- bins {w}s: dispersion index (var/mean)')
    cnt = {k: np.bincount((v.to_numpy() // w).astype(int), minlength=T // w)[:T // w] for k, v in streams.items()}
    for k, c in cnt.items(): print(f'  {k:11s} mean {c.mean():9.2f} DI {c.var() / c.mean():7.2f}')
    if w in (10, 60):
        M = pd.DataFrame(cnt); print('  corr\n', M.corr().round(3).to_string())
cnt60 = pd.DataFrame({k: np.bincount((v.to_numpy() // 60).astype(int), minlength=100)[:100] for k, v in streams.items()}) / 60
cnt60.index = cnt60.index * 60
pd.set_option('display.width', 200)
print((cnt60).round(2).to_string())
cnt60.to_csv(Path(__file__).parent / 'rates_60s.csv')
