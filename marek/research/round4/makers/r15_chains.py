"""Chains of the 15-lot agent: a 15-lot maker insert placed exactly 1 us after the end (fill or cancel row) of an
earlier 15-lot maker order. Writes cache/r15_chains.pkl (orders with r15 chain id / position)."""
import numpy as np, pandas as pd
from prep import load
o = load().reset_index(drop=True)
mk = (~o.ioc) & (o.volume == 15)
q = o[mk].copy()
print('15-lot maker inserts', len(q))
endt = q.end_t_us.to_numpy().astype(np.int64)
t = q.t_us.to_numpy()
idx = q.index.to_numpy()
# predecessor: 15-lot order whose end_t_us == t - 1
s = pd.Series(idx, index=endt)
s = s[~s.index.duplicated()]
pred = s.reindex(t - 1).to_numpy()
pred = np.where(np.isnan(pred), -1, pred).astype(np.int64)
q['pred'] = pred
# also: inserted 1 us after any IOC taker (trigger even if no 15 pred)
tt = set(o.t_us[o.ioc].to_numpy().tolist())
q['after_taker'] = [x - 1 in tt for x in t]
allend = pd.Series(1, index=o.end_t_us.dropna().astype(np.int64).to_numpy())
q['after_anyend'] = np.isin(t - 1, o.end_t_us.dropna().astype(np.int64).to_numpy())
print('with 15-lot predecessor', (pred >= 0).sum(), ' after taker', q.after_taker.sum(), ' after any end', q.after_anyend.sum())
# chain ids
cid = {}
chain = np.empty(len(q), np.int64); k = np.zeros(len(q), np.int64)
pos = {ix: n for n, ix in enumerate(idx)}
for n, (ix, p) in enumerate(zip(idx, pred)):
    if p >= 0 and p in pos:
        m = pos[p]; chain[n] = chain[m]; k[n] = k[m] + 1
    else:
        chain[n] = ix
q['chain'] = chain; q['k'] = k
q['L'] = q.groupby('chain').k.transform('max') + 1
# successor exists?
q['has_succ'] = q.index.isin(q.pred[q.pred >= 0])
print('chain length dist (orders):'); print(q[q.k == 0].L.value_counts().sort_index().to_string())
q.to_pickle('cache/r15_chains.pkl')
c = q[q.L >= 2]
print('orders in chains L>=2', len(c), 'chains', c.chain.nunique())
g = c.groupby('chain').agg(t0=('t_us', 'min'), t1=('end_t_us', 'max'), n=('t_us', 'size'), sg=('sg', 'mean'),
                           filled=('filled', 'sum'))
g['t0'] /= 1e6; g['t1'] /= 1e6; g['dur'] = g.t1 - g.t0
print(g.sort_values('t0').to_string())
