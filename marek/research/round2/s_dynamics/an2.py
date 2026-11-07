import numpy as np, pandas as pd, sys
import prep
d = prep.build(); tag = sys.argv[1] if len(sys.argv) > 1 else 'v1'
r = dict(np.load(prep.common.CACHE / f'smooth_{tag}.npz'))
t = d['t']; m = r['mean']
k = d['kind']; p = d['price']
tk = np.where((k == 4) | (k == 5))[0]
x = d['sign'][tk] * (p[tk] - m[tk])
df = pd.DataFrame(dict(t=t[tk], x=x, vol=d['volume'][tk], sd=r['sd'][tk], buy=d['sign'][tk] > 0))
df['out'] = df.x > 1.5
# local realized vol of S: |S(t+1s)-S(t-1s)|
def S_at(tt):
    i = np.clip(np.searchsorted(t, tt) - 1, 0, len(t) - 1); return m[i]
df['mv_past'] = np.abs(S_at(df.t) - S_at(df.t - 2))
df['rv'] = pd.Series(np.abs(np.diff(S_at(np.arange(0, 6001, 1.0))))).rolling(10, center=True, min_periods=1).mean().to_numpy()[np.minimum(df.t.astype(int), 5999)]
print(df.groupby(pd.cut(df.t, np.arange(0, 6001, 300))).agg(n=('x', 'size'), out=('out', 'mean'), rv=('rv', 'mean')).round(3).to_string())
print(df.groupby(pd.qcut(df.rv, 10)).out.mean().round(3))
print(df.groupby(pd.qcut(df.vol, 5)).out.mean().round(3))
print('outlier x hist (ceil of x-1.5):', np.bincount(np.minimum(np.ceil(df.x[df.out] - 1.5).astype(int), 40)))
# x fractional structure: p - round(S+-1) offset in integer ticks
off_int = d['sign'][tk] * (p[tk] - np.round(m[tk] + d['sign'][tk]))
print('integer extra offset counts', pd.Series(off_int).value_counts().sort_index().head(30).to_dict())
