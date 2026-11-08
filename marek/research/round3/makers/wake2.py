"""Calm, same-side touch moves 1 toward the ladder and then stays >= 1 s: how many inserts land at the new touch-7 level
in that second, their timing order and sizes; also the count of live orders already at that level (from all makers)."""
import numpy as np, pandas as pd
from collections import Counter
o = pd.read_pickle('cache/chains.pkl')
g = np.load('cache/regime_sz.npz')['post'][:, 1]
m = o[~o.ioc.astype(bool)].copy(); m['end_t'] = m.end_t_us.fillna(6e9)
q = pd.read_pickle('cache/quotes.pkl')
for side, col in ((1, 'bid'), (-1, 'ask')):
    x = q[['t_us', col]].dropna(); x = x[x[col].diff() != 0]
    tt = x.t_us.to_numpy(); px = x[col].to_numpy(); mv = np.r_[0, np.diff(px)] * side; nxt = np.r_[tt[1:], 6e9]
    ms = m[m.sg == side]; mt = ms.t_us.to_numpy(); mp = ms.price.to_numpy(); me = ms.end_t.to_numpy(); mvv = ms.volume.to_numpy()
    idx = np.flatnonzero((mv == -1) & (nxt - tt > 1e6) & (g[(tt // 1_000_000).astype(int)] < .5))
    N = Counter(); pre = Counter(); first = []; sz = []
    for i in idx:
        lvl = px[i] - side * 7; t0 = tt[i]
        j0, j1 = np.searchsorted(mt, t0), np.searchsorted(mt, t0 + 1e6)
        hit = np.flatnonzero(mp[j0:j1] == lvl); N[len(hit)] += 1
        jj = np.searchsorted(mt, t0 - 120e6)
        pre[int(((mp[jj:j0] == lvl) & (me[jj:j0] > t0)).sum())] += 1
        if len(hit): first.append((mt[j0 + hit[0]] - t0) / 1e3); sz += list(mvv[j0:j1][hit])
    print('side', side, 'events', len(idx), 'inserts per event at new -7 within 1 s:', sorted(N.items()))
    print('   live orders already at that level at the move:', sorted(pre.items()))
    print('   first-insert delay ms quantiles', np.percentile(first, [10, 25, 50, 75, 90]).round(1), ' sizes q', np.percentile(sz, [10, 50, 90]))
