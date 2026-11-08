"""Ladder cancels: within-burst cancel synchrony; what precedes the first cancel of a burst
(same-side burst start, trade, touch move)."""
import numpy as np, pandas as pd
o = pd.read_pickle('cache/chains.pkl')
g = np.load('cache/regime_sz.npz')['post']
o['vol'] = g[(o.t_us//1_000_000).to_numpy(), 1] > .5
lad = o[(o.L >= 2) & (o.L <= 6) & ~o.ioc.astype(bool)].copy()
lad['canc'] = lad.end == 'cancelled'
c = lad[lad.canc]
sp = c.groupby('chain').cancel_us.agg(['min','max','size'])
sp = sp[sp['size'] >= 2]
d = (sp['max'] - sp['min'])/1e3
print('cancelled-member spread (ms) quantiles, chains with >=2 cancelled:', np.percentile(d, [10,25,50,75,90]).round(1))
print('share spread < 50 ms', (d < 50).mean(), '< 100', (d<100).mean())
# pairwise gaps between consecutive cancel rows in a chain, sorted by rung
x = c.sort_values(['chain','k'])
x['dc'] = x.groupby('chain').cancel_us.diff()/1e3
print('consecutive-rung cancel gaps ms quantiles', np.nanpercentile(x.dc, [5,10,25,50,75,90,95]).round(2))
print('exactly 2 ms?', np.histogram(x.dc.dropna(), bins=[-1e9,-100,-10,-2.5,-1.5,0,1.5,2.5,10,100,1e9])[0])
# cancel rows exact spacing
cu = np.sort(o.cancel_us.dropna().to_numpy())
dd = np.diff(cu)
for k in [1999,2000,2001, 1000, 0, 1]:
    print('cancel-row gap', k, (dd == k).sum())
# precedents: time from chain's first cancel back to the latest same-side burst start
st = lad[lad.k == 0][['t_us','sg','chain']].sort_values('t_us')
for s in (1, -1):
    ss = st[st.sg == s].t_us.to_numpy()
    cc = sp.join(lad[lad.k==0].set_index('chain')[['sg','t_us','vol']])
    cc = cc[cc.sg == s]
    j = np.searchsorted(ss, cc['min'].to_numpy()) - 1
    prev = ss[np.maximum(j, 0)]
    gap = (cc['min'].to_numpy() - prev)/1e3
    own = prev == cc.t_us.to_numpy()
    print('side', s, 'gap from latest same-side burst start to first cancel (ms), excluding own:', np.percentile(gap[~own], [5,10,25,50,75]).round(1), 'own-latest share', own.mean())
