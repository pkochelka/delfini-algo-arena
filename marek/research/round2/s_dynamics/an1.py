import numpy as np, pandas as pd, sys
import prep
d = prep.build(); tag = sys.argv[1] if len(sys.argv) > 1 else 'v1'
r = dict(np.load(prep.common.CACHE / f'smooth_{tag}.npz'))
t = d['t']; m = r['mean']; sd = r['sd']
print('post sd pct', np.percentile(sd, [10, 50, 90, 99, 99.9]).round(3))
k = d['kind']; p = d['price']
# takers offset
tk = (k == 4) | (k == 5)
x = d['sign'][tk] * (p[tk] - m[tk])
print('taker sign*(p-S) hist:')
h = np.histogram(x, bins=[-5, -0.5, 0, 0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 4, 6, 10, 20, 50, 200])
print(list(zip(h[1][:-1], h[0])))
inr = (x >= 0.5) & (x <= 1.5)
print('share in [0.5,1.5]', inr.mean())
# maker gap
mk = k <= 3
gap = np.where(d['sign'][mk] > 0, m[mk] - p[mk], p[mk] - m[mk])
print('maker gap S-bid pct', np.percentile(gap, [0.01, 0.1, 1, 5, 50, 95]).round(3), 'viol(<0.05)', (gap < 0.05).mean(), 'viol(<0)', (gap < 0).mean())
