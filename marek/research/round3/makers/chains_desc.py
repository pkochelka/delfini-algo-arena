"""Describe ladder chains: side, price step, sizes, lifetimes, cancel synchrony, regime, placement vs touch."""
import numpy as np, pandas as pd
o = pd.read_pickle('cache/chains.pkl')
g = np.load('cache/regime_sz.npz')['post']
o['vol'] = g[(o.t_us//1_000_000).to_numpy(), 1] > .5
o['life'] = (o.cancel_us - o.t_us)/1e6
o['rel'] = np.where(o.sg>0, o.price - o.bid, o.ask - o.price)
ch = o[(o.L >= 2) & (o.L <= 6) & ~o.ioc.astype(bool)].copy()
base = o[(o.L == 1) & ~o.ioc.astype(bool)]
p = ch.set_index('chain')
# side consistency + price step
prev = o.loc[ch.pred[ch.k>0].to_numpy()]
cur = ch[ch.k>0]
print('same side as pred', (prev.sg.to_numpy() == cur.sg.to_numpy()).mean())
step = (cur.price.to_numpy() - prev.price.to_numpy()) * cur.sg.to_numpy()   # >0 = toward/through touch
print('price step (sg-signed) dist', pd.Series(step).value_counts().head(10).to_dict())
print('vol share: chains', ch.vol.mean(), 'base', base.vol.mean())
print('chain sizes q', np.percentile(ch.volume, [1,5,10,25,50,75,90,95,99]))
print('logsize mean sd', np.log(ch.volume).mean(), np.log(ch.volume).std())
for L in range(2, 7):
    s = ch[ch.L == L]
    print(f'L={L} n_chains={s.chain.nunique()} med size {s.volume.median()} vol share {s.vol.mean():.3f} life mean {s.life.mean():.2f} med {s.life.median():.2f}')
print('by k: size median, rel median, rel dist')
for k in range(6):
    s = ch[ch.k == k]
    print(k, len(s), s.volume.median(), s.rel.value_counts(normalize=True).sort_index().loc[-8:2].round(3).to_dict())
print('base rel dist', base.rel.value_counts(normalize=True).sort_index().loc[-8:2].round(3).to_dict())
print('lifetime quantiles chain', np.nanpercentile(ch.life, [10,25,50,75,90,99]), 'base', np.nanpercentile(base.life,[10,25,50,75,90,99]))
# cancel synchrony within chain
cs = ch.groupby('chain').cancel_us.agg(lambda x: (x.max()-x.min())/1e3)
print('within-chain cancel spread ms quantiles', np.percentile(cs, [10,25,50,75,90]))
fill = ch.filled > 0
print('filled share chain', fill.mean(), 'base', (base.filled>0).mean())
# vol vs calm chains
for v in (False, True):
    s = ch[ch.vol == v]
    print('vol' if v else 'calm', 'n', len(s), 'size q', np.percentile(s.volume, [10,50,90]), 'life med', s.life.median(), 'rel med', s.rel.median())
