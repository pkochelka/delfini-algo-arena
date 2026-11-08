"""Reconstruct ladders: inserts linked by an exact 2000 us gap to an earlier insert. Writes cache/chains.pkl
(orders with chain id, position in chain, chain length)."""
import numpy as np, pandas as pd
from prep import load
o = load().reset_index(drop=True)
t = o.t_us.to_numpy()
# map exact time -> index (times are near-unique)
vals, first_idx, cnt = np.unique(t, return_index=True, return_counts=True)
print('dup timestamps', (cnt > 1).sum())
pos = np.searchsorted(vals, t - 2000)
has = (pos < len(vals)) & (vals[np.minimum(pos, len(vals)-1)] == t - 2000)
pred = np.where(has, first_idx[np.minimum(pos, len(vals)-1)], -1)
chain = np.arange(len(o)); k = np.zeros(len(o), int)
for i in range(len(o)):
    p = pred[i]
    if p >= 0:
        chain[i] = chain[p]; k[i] = k[p] + 1
o['pred'] = pred; o['chain'] = chain; o['k'] = k
L = o.groupby('chain').k.transform('max') + 1
o['L'] = L
print('orders in chains (L>=2):', (L >= 2).sum(), 'ioc among them', o[L>=2].ioc.sum())
c = o[o.k == 0]
print('chain length distribution:'); print(c.L.value_counts().sort_index().head(20))
# random-coincidence baseline: gaps of 1999/2001 analogous
o.to_pickle('cache/chains.pkl')
