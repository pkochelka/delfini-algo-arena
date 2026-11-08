"""Ladder size law: per-burst level (mean log size) distribution by regime; within-burst dispersion and shape;
dependence on rung k, length L, side."""
import numpy as np, pandas as pd
from scipy import stats
o = pd.read_pickle('cache/chains.pkl')
g = np.load('cache/regime_sz.npz')['post']
o['vol'] = g[(o.t_us//1_000_000).to_numpy(), 1] > .5
lad = o[(o.L >= 2) & (o.L <= 6) & ~o.ioc.astype(bool)].copy()
lad['ls'] = np.log(lad.volume)
bm = lad.groupby('chain').ls.transform('mean')
lad['w'] = lad.ls - bm
for v in (False, True):
    s = lad[lad.vol == v]
    b = s.groupby('chain').agg(m=('ls','mean'), L=('L','first'))
    print('VOL' if v else 'CALM', 'bursts', len(b))
    print('  burst mean log size: mean %.3f sd %.3f skew %.2f kurt %.2f' % (b.m.mean(), b.m.std(), stats.skew(b.m), stats.kurtosis(b.m)))
    print('  quantiles of exp(burst mean)', np.exp(np.percentile(b.m, [1,5,25,50,75,95,99])).round(1))
    # within: unbiased sd of log size within burst
    ss = s.groupby('chain').ls.agg(['var','size'])
    wv = (ss['var']*(ss['size']-1)).sum() / (ss['size']-1).sum()
    print('  pooled within-burst log sd %.3f' % np.sqrt(wv))
    # ratio of rung sizes to burst median: check uniform multiplicative noise
    r = s.volume / s.groupby('chain').volume.transform('mean')
    print('  size/burst-mean quantiles', np.percentile(r, [0.5,1,5,25,50,75,95,99,99.5]).round(3))
    print('  mean log size by k', s.groupby('k').ls.mean().round(3).to_dict())
    print('  burst mean by L', b.groupby('L').m.mean().round(3).to_dict())
    print('  by side', s.groupby('sg').ls.mean().round(3).to_dict())
    # 2-rung bursts: distribution of log ratio
    two = s[s.L == 2].sort_values(['chain','k'])
    lr = two.groupby('chain').ls.agg(lambda x: x.iloc[1]-x.iloc[0])
    print('  L=2 log ratio rung1/rung0: sd %.3f, quantiles' % lr.std(), np.percentile(lr, [1,5,25,50,75,95,99]).round(3), 'kurt %.2f' % stats.kurtosis(lr))
