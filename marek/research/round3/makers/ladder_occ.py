"""At each ladder (chain) start: which levels (relative to same-side touch) already hold a live ladder order of the
same side, and which levels does the chain add?"""
import numpy as np, pandas as pd
from collections import Counter
o = pd.read_pickle('cache/chains.pkl')
g = np.load('cache/regime_sz.npz')['post']
o['vol'] = g[(o.t_us//1_000_000).to_numpy(), 1] > .5
lad = o[(o.L >= 2) & (o.L <= 6) & ~o.ioc.astype(bool)].copy()
lad['end_t'] = lad.end_t_us.fillna(6e9)
t = lad.t_us.to_numpy(); e = lad.end_t.to_numpy(); s = lad.sg.to_numpy(); pr = lad.price.to_numpy()
starts = lad[lad.k == 0]
pat = Counter(); full = Counter(); n = 0
for _, r in starts.iterrows():
    t0 = r.t_us; tb = r.bid if r.sg > 0 else r.ask
    if np.isnan(tb): continue
    j0 = np.searchsorted(t, t0 - 60e6); j1 = np.searchsorted(t, t0)
    sl = slice(j0, j1)
    live = (e[sl] > t0) & (s[sl] == r.sg)
    lv = sorted(set(((pr[sl][live] - tb) * r.sg).astype(int)))
    mine = lad[(lad.chain == r.chain)]
    nw = sorted(((mine.price - tb) * r.sg).astype(int))
    key = (r.vol, tuple(lv), tuple(nw))
    pat[key] += 1; full[(r.vol, tuple(sorted(set(lv) | set(nw))))] += 1
    n += 1
    if n >= 6000: break
print('top patterns (vol, live levels, new levels):')
for k, c in pat.most_common(40): print(c, k)
print('union patterns:')
for k, c in full.most_common(25): print(c, k)
