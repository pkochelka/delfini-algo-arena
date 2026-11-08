"""Compare size pmf in episode minutes vs normal minutes; excess histogram."""
import numpy as np, pandas as pd
from prep import load
o = load(); m = o[~o.ioc.astype(bool)].copy()
m['min'] = (m.t_us // 60_000_000).astype(int)
ep = {10,11,12,13,38,39,40,41,63,64,66,68,92,93,96,97,98}
calm = set(range(100)) - ep - {17,19,20,26,31,33,48,54,57,58,62,72,78,80,65}
E = m[m['min'].isin(ep)].volume.to_numpy(); C = m[m['min'].isin(calm)].volume.to_numpy()
nE, nC = len(ep), len(calm)
print('rate/min ep', len(E)/nE, 'calm', len(C)/nC)
edges = np.unique(np.round(np.exp(np.arange(0, 12, 0.25)))).astype(int)
hE, _ = np.histogram(E, edges); hC, _ = np.histogram(C, edges)
ex = hE/nE - hC/nC
for a,b,he,hc,x in zip(edges[:-1], edges[1:], hE/nE, hC/nC, ex):
    print(f'{a:7d}-{b:7d} ep {he:8.1f} calm {hc:8.1f} excess {x:8.1f}')
print('excess total', ex.sum())
# exact small values
for v in range(1, 60):
    print(v, round((E==v).sum()/nE,1), round((C==v).sum()/nC,1), end=' | ')
print()
o2 = pd.read_pickle('../../round2/makers/cache/orders.pkl'); m2 = o2[~o2.ioc.astype(bool)]
l2 = np.log(m2.volume); lc = np.log(C)
print('r2 logsize mean sd skew kurt', l2.mean(), l2.std(), pd.Series(l2).skew(), pd.Series(l2).kurt())
print('r3 calm-min logsize', lc.mean(), lc.std(), pd.Series(lc).skew(), pd.Series(lc).kurt())
print('r2 quantiles', np.percentile(m2.volume, [1,5,10,25,50,75,90,95,99]))
print('r3 calm quantiles', np.percentile(C, [1,5,10,25,50,75,90,95,99]))
