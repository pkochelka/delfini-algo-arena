import numpy as np, pandas as pd, sys
import prep
d = prep.build(); tag = sys.argv[1] if len(sys.argv) > 1 else 'v1'
r = dict(np.load(prep.common.CACHE / f'smooth_{tag}.npz'))
t = d['t']; m = r['mean']
k = d['kind']; p = d['price']
tk = np.where(((k == 4) | (k == 5)) & (r['sd'] < 0.15))[0]
x = d['sign'][tk] * (p[tk] - m[tk])
for a, b in [(600, 1500), (3600, 3900), (5400, 5700), (1800, 2100)]:
    mm = (t[tk] >= a) & (t[tk] < b)
    h = np.histogram(x[mm], bins=np.arange(0, 12.01, 0.25))[0]
    print(a, b, mm.sum(), ' '.join(map(str, h)))
