"""Within-burst size noise shape: L=6 bursts, ratios to burst mean, min/max ratio; vs U(0.8,1.2) simulation."""
import numpy as np, pandas as pd
from scipy import stats
o = pd.read_pickle('cache/chains.pkl')
g = np.load('cache/regime_sz.npz')['post']
o['vol'] = g[(o.t_us//1_000_000).to_numpy(), 1] > .5
lad = o[(o.L >= 2) & (o.L <= 6) & ~o.ioc.astype(bool)].copy()
lad = lad[lad.groupby('chain').vol.transform('nunique') == 1]
lad = lad[(lad.groupby('chain').volume.transform('mean') > 60) & (lad.groupby('chain').k.transform('size') == lad.L)]   # large sizes: rounding negligible
rng = np.random.default_rng(1)
for L in (2, 4, 6):
    s = lad[lad.L == L]
    v = s.sort_values(['chain','k']).volume.to_numpy().reshape(-1, L).astype(float)
    mm = v.max(1)/v.min(1)
    r = (v / v.mean(1, keepdims=True)).ravel()
    B = v.mean(1)
    for name, sim in [('U(.8,1.2)', lambda n: rng.uniform(.8, 1.2, (n, L))), ('LN(0,.115)', lambda n: np.exp(rng.normal(0, .115, (n, L)))),
                      ('U(.75,1.25)', lambda n: rng.uniform(.75, 1.25, (n, L))), ('U(.9,1.1)', lambda n: rng.uniform(.9, 1.1, (n, L)))]:
        x = np.round(B[:, None] * sim(len(v)))
        mms = x.max(1)/x.min(1); rs = (x / x.mean(1, keepdims=True)).ravel()
        print(f'L={L} n={len(v)} {name:12s} max/min q50,90,99 obs', np.percentile(mm, [50,90,99]).round(3), 'sim', np.percentile(mms, [50,90,99]).round(3),
              ' r q1,5,95,99 obs', np.percentile(r, [1,5,95,99]).round(3), 'sim', np.percentile(rs, [1,5,95,99]).round(3))
    print('  kurtosis of r obs', stats.kurtosis(r).round(2))
