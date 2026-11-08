"""S drift rate during the last order of each instance vs during earlier orders, and last-order lifetime."""
import numpy as np, pandas as pd
S = pd.read_parquet('../s_dynamics/s_path.parquet')
st = S.t_us.to_numpy(); sh = S.s_hat.to_numpy()
def s(t): return sh[np.clip(np.searchsorted(st, t, 'right') - 1, 0, len(st) - 1)]
a = pd.read_pickle('cache/r15_v2.pkl').sort_values('t_us')
a['dS'] = a.sg * (s(a.end_t_us.to_numpy()) - s(a.t_us.to_numpy()))
a['life'] = (a.end_t_us - a.t_us) / 1e6
a['last'] = a.k == a.L - 1
a['first'] = a.k == 0
for lab, m in [('first', a['first']), ('middle', ~a['first'] & ~a['last']), ('last', a['last']), ('last, filled', a['last'] & (a.end == 'filled'))]:
    x = a[m]
    print(f'{lab:13s} n {len(x):4d} life mean {x.life.mean():.3f} med {x.life.median():.3f}  dS sum/life sum {x.dS.sum() / x.life.sum():.2f} tick/s, mean dS {x.dS.mean():.2f}')
# second-to-last
a['k_from_end'] = a.L - 1 - a.k
print(a.groupby('k_from_end').apply(lambda x: pd.Series({'n': len(x), 'rate': x.dS.sum() / x.life.sum(), 'life': x.life.mean()})).head(8).round(2).T.to_string())
print(a.groupby('k').apply(lambda x: pd.Series({'n': len(x), 'rate': x.dS.sum() / x.life.sum(), 'life': x.life.mean()})).head(10).round(2).T.to_string())
