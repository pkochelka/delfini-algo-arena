"""Compare round-3 calm-minute size pmf with round 2 (per 0.25 log bins and per-integer around 60-300)."""
import numpy as np, pandas as pd
from prep import load
o = load(); m = o[~o.ioc.astype(bool)].copy()
m['min'] = (m.t_us // 60_000_000).astype(int)
ep = {10,11,12,13,38,39,40,41,63,64,66,68,92,93,96,97,98}
calm = set(range(100)) - ep - {17,19,20,26,31,33,48,54,57,58,62,72,78,80,65}
C = m[m['min'].isin(calm)].volume.to_numpy(); E = m[m['min'].isin(ep)].volume.to_numpy()
o2 = pd.read_pickle('../../round2/makers/cache/orders.pkl'); R = o2[~o2.ioc.astype(bool)].volume.to_numpy()
edges = np.unique(np.round(np.exp(np.arange(0, 12, 0.25)))).astype(int)
hC = np.histogram(C, edges)[0]/len(C)*1e4; hR = np.histogram(R, edges)[0]/len(R)*1e4; hE = np.histogram(E, edges)[0]/len(E)*1e4
for a,b,c,r,e in zip(edges[:-1], edges[1:], hC, hR, hE):
    if a < 5000: print(f'{a:6d}-{b:6d}  r2 {r:7.1f}  r3calm {c:7.1f}  r3ep {e:7.1f}  ratio c/r {c/r:5.2f}')
