import numpy as np, pandas as pd
from prep import load
o = load(); m = o[~o.ioc.astype(bool)].copy().reset_index(drop=True)
t = m.t_us.to_numpy()
d = np.diff(t)
for k in [1998,1999,2000,2001,2002, 999,1000,1001, 2999,3000,3001, 4000, 5000]:
    print(k, (d==k).sum())
# all exact-multiple-of-1000 gaps between any pair within 10 rows
cnt = {}
for lag in range(1, 8):
    dd = t[lag:] - t[:-lag]
    vals, c = np.unique(dd[dd < 20000], return_counts=True)
    top = np.argsort(c)[-5:]
    print('lag', lag, list(zip(vals[top], c[top])))
