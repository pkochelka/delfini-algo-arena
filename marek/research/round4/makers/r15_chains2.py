"""15-lot agent chains, v2. An agent order is a maker insert placed exactly 1 us after the end row (fill or cancel)
of an earlier agent order on the same side, with volume 15 if that order was fully filled, or its unfilled remainder
if it was cancelled. Seeds: any 15-lot maker order. Writes cache/r15_v2.pkl: maker orders in chains of length >= 2,
with chain, k, L, and 'link' (fill / cancel)."""
import numpy as np, pandas as pd
from collections import defaultdict
from prep import load
o = load().reset_index(drop=True)
m = o[~o.ioc]
t = m.t_us.to_numpy(); sg = m.sg.to_numpy(); vol = m.volume.to_numpy(); fil = m.filled.to_numpy()
end = m.end.to_numpy(); endt = m.end_t_us.fillna(-1).to_numpy().astype(np.int64); idx = m.index.to_numpy()
# index of inserts by time
by_t = pd.Series(np.arange(len(t)), index=t)
chain = np.full(len(t), -1); k = np.zeros(len(t), int); link = np.array([''] * len(t), dtype=object)
for n in range(len(t)):
    if chain[n] < 0:
        if vol[n] != 15:
            continue
        chain[n] = n
    # successor
    if endt[n] < 0:
        continue
    s = by_t.get(endt[n] + 1)
    if s is None or chain[s] >= 0 or sg[s] != sg[n]:
        continue
    exp = 15 if end[n] == 'filled' else vol[n] - fil[n]
    if vol[s] != exp:
        continue
    chain[s] = chain[n]; k[s] = k[n] + 1; link[s] = 'fill' if end[n] == 'filled' else 'cancel'
mm = m.copy(); mm['chain'] = chain; mm['k'] = k; mm['link'] = link
mm = mm[mm.chain >= 0]
mm['L'] = mm.groupby('chain').k.transform('max') + 1
a = mm[mm.L >= 2].copy()
print('agent orders (L>=2)', len(a), 'chains', a.chain.nunique(), 'non-15 orders', (a.volume != 15).sum())
print('links', a.link.value_counts().to_dict())
print('chain length:', a[a.k == 0].L.value_counts().sort_index().to_dict())
g = a.groupby('chain').agg(t0=('t_us', 'min'), t1=('end_t_us', 'max'), n=('t_us', 'size'), sg=('sg', 'first'),
                           filled=('filled', 'sum'), lastend=('end', 'last'))
g['t0'] /= 1e6; g['t1'] /= 1e6
print(g.sort_values('t0').to_string())
print('filled-sum mod 15 == 0 share', (g.filled % 15 == 0).mean(), g.lastend.value_counts().to_dict())
a.to_pickle('cache/r15_v2.pkl')
