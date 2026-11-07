import numpy as np, pandas as pd, sys
import prep
d = prep.build(); tag = sys.argv[1] if len(sys.argv) > 1 else 'v1'
r = dict(np.load(prep.common.CACHE / f'smooth_{tag}.npz'))
t = d['t']; m = r['mean']
k = d['kind']; p = d['price']
tk = np.where((k == 4) | (k == 5))[0]
sg = d['sign'][tk]
x = sg * (p[tk] - m[tk])
def S_at(tt):
    i = np.clip(np.searchsorted(t, tt, 'right') - 1, 0, len(t) - 1); return m[i]
res = pd.DataFrame(dict(x=x, xi=np.round(x - 1)))
for lag in [-0.5, -0.1, -0.02, 0.005, 0.02, 0.1, 0.5, 2]:
    res[f'dS{lag}'] = sg * (S_at(t[tk] + lag) - m[tk]) if lag > 0 else sg * (m[tk] - S_at(t[tk] + lag))
res['filled'] = d['filled'][tk]; res['vol'] = d['volume'][tk]
print(res.groupby(res.xi.clip(-1, 12)).mean().round(2).to_string())
