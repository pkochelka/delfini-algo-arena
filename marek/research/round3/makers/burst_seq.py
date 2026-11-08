"""Sequential structure of bursts on a side: does a burst start where the previous same-side burst ended?"""
import numpy as np, pandas as pd
o = pd.read_pickle('cache/chains.pkl')
g = np.load('cache/regime_sz.npz')['post']
o['vol'] = g[(o.t_us//1_000_000).to_numpy(), 1] > .5
o['rel'] = np.where(o.sg>0, o.price - o.bid, o.ask - o.price)
lad = o[(o.L >= 2) & (o.L <= 6) & ~o.ioc.astype(bool)]
b = lad.groupby('chain').agg(t=('t_us','first'), sg=('sg','first'), p0=('price','first'), plast=('price','last'),
                             r0=('rel','first'), rl=('rel','last'), L=('L','first'), vol=('vol','first'),
                             touch=('bid','first'), ask=('ask','first'), B=('volume','mean')).sort_values('t')
b['touch'] = np.where(b.sg > 0, b.touch, b.ask)
for s in (1, -1):
    x = b[b.sg == s].copy()
    x['prev_end_abs'] = x.plast.shift(); x['prev_t'] = x.t.shift(); x['prev_rl'] = x.rl.shift(); x['prev_r0'] = x.r0.shift(); x['prevB'] = x.B.shift()
    x['prev_end_rel_now'] = (x.prev_end_abs - x.touch) * s     # previous end, relative to current touch
    x['dt'] = (x.t - x.prev_t)/1e3
    v = x[x.vol]
    print('side', s, 'VOL: crosstab start rel vs prev end rel (rel to current touch)')
    print(pd.crosstab(v.r0.clip(-7,-1), v.prev_end_rel_now.clip(-9, 0)))
    print('dt (ms) between consecutive same-side bursts quantiles', np.percentile(v.dt.dropna(), [5,10,25,50,75,90]).round(1))
    print('corr log B consecutive', np.corrcoef(np.log(v.B.iloc[1:]), np.log(v.prevB.iloc[1:]))[0,1].round(3))
    print(pd.crosstab(v.r0.clip(-7,-1), pd.cut(v.dt, [0, 5, 10, 20, 50, 100, 200, 500, 1e9])))
    break
