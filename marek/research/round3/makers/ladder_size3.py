"""Burst size level B per regime from clean bursts (L>=3, single regime), lognormal fit + quantiles; rung sizes vs
U(0.75,1.25)*B. Also B vs time within a regime segment (drift?)."""
import numpy as np, pandas as pd
from scipy import stats
o = pd.read_pickle('cache/chains.pkl')
g = np.load('cache/regime_sz.npz')['post']
o['pv'] = g[(o.t_us//1_000_000).to_numpy(), 1]
lad = o[(o.L >= 3) & (o.L <= 6) & ~o.ioc.astype(bool)].copy()
b = lad.groupby('chain').agg(m=('volume', lambda x: np.log(x.mean())), pv=('pv','mean'), t=('t_us','first'), L=('L','first'))
for name, s in [('calm', b[b.pv < .05]), ('vol', b[b.pv > .95])]:
    x = s.m
    print(f'{name}: n={len(s)} log B mean {x.mean():.3f} sd {x.std():.3f} skew {stats.skew(x):.2f} kurt {stats.kurtosis(x):.2f}  median B {np.exp(x.median()):.1f}')
    print('   B quantiles', np.exp(np.percentile(x, [1,5,10,25,50,75,90,95,99])).round(1))
    # trimmed
    xt = x[(x > x.quantile(.02))]
    print(f'   trimmed 2% low: mean {xt.mean():.3f} sd {xt.std():.3f} skew {stats.skew(xt):.2f}')
    # uniform in log? check range
    print('   hist of log B:', np.histogram(x, np.arange(1.5, 7.01, 0.25))[0].tolist())
