"""Preliminary: ladder rung prices vs S (s_dynamics smooth_v2r2, round-2 params, contaminated by ladders) and vs touch."""
import sys, numpy as np, pandas as pd
tag = sys.argv[1] if len(sys.argv) > 1 else 'smooth_v2r2'
z = np.load(f'../s_dynamics/_cache/{tag}.npz')
o = pd.read_pickle('cache/chains.pkl')
o['S'] = z['mean']; o['Ssd'] = z['sd']
g = np.load('cache/regime_sz.npz')['post']
o['vol'] = g[(o.t_us//1_000_000).to_numpy(), 1] > .5
o['rel'] = np.where(o.sg>0, o.price - o.bid, o.ask - o.price)
o['dS'] = (o.S - o.price) * o.sg          # distance behind S (>0 = behind)
o['tS'] = np.where(o.sg>0, o.S - o.bid, o.ask - o.S)   # touch distance behind S
lad = o[(o.L >= 2) & (o.L <= 6) & ~o.ioc.astype(bool)]
st = lad[lad.k == 0]
for v in (False, True):
    s = st[st.vol == v]
    print('VOL' if v else 'CALM')
    print(' start dS quantiles', np.percentile(s.dS, [5,10,25,50,75,90,95]).round(2))
    print(' by start rel: mean tS (touch behind S), mean dS')
    print(s.groupby(s.rel.clip(-7, 0)).agg(n=('dS','size'), tS=('tS','mean'), dS=('dS','mean'), dS_sd=('dS','std')).round(2))
    print(' frac(S)-based: start dS by floor  ->', pd.cut(s.dS, np.arange(-1, 9, 0.5)).value_counts().sort_index().to_dict())
