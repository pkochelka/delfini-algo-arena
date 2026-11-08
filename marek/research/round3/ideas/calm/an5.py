"""Pinned episodes in calm: durations, how they end, and what the reducing-side quote looks like meanwhile."""
import sys, pickle, pathlib, numpy as np, pandas as pd
R3 = pathlib.Path(__file__).resolve().parents[2]
L = pickle.load(open(sys.argv[1], 'rb'))
sm = pd.DataFrame(L['samp'], columns='t pos turb F sd bb ba bq aq obp obq oba oap oaq oaa'.split())
G = np.load(R3 / 'ideas/calm/Y_grid.npy')
i = np.clip(np.searchsorted(G[:, 0], sm.t.values * 1e-6) - 1, 0, len(G) - 1)
sm['g'] = G[i, 2]
c = sm[sm.g == 0].copy()
pin = c.pos.abs() >= 100
# reducing side quote while pinned: long -> our ask; short -> our bid
long_ = c.pos >= 100; short = c.pos <= -100
ra = c[long_]; rb = c[short]
print('pinned share', pin.mean(), 'long', long_.mean(), 'short', short.mean())
print('long pinned: have ask', (ra.oaq > 0).mean(), 'ask at touch', (ra.oap == ra.ba).mean(), 'ask behind', (ra.oap > ra.ba).mean(),
      'mkt ahead of our ask (median, when at touch)', ra.oaa[ra.oap == ra.ba].median(), 'ask dist F', (ra.oap - ra.F).mean())
print('short pinned: have bid', (rb.obq > 0).mean(), 'bid at touch', (rb.obp == rb.bb).mean(), 'bid behind', (rb.obp < rb.bb).mean(),
      'mkt ahead (median, at touch)', rb.oaa.median() if False else rb.obq.median())
print('unpinned: bid at touch', (c[~pin].obp == c[~pin].bb).mean(), 'ask at touch', (c[~pin].oap == c[~pin].ba).mean())
print('ask - F when long pinned: ', (ra.oap - ra.F).describe().round(3).to_dict())
print('F - bb when long pinned:', (ra.F - ra.bb).describe().round(3).to_dict())
# episode durations
t = c.t.values * 1e-6; p = pin.values
starts = np.flatnonzero(p & ~np.r_[False, p[:-1]]); ends = np.flatnonzero(p & ~np.r_[p[1:], False])
d = (t[ends] - t[starts]) + 0.05
d = d[d < 60]
print('pinned episodes', len(d), 'dur median', np.median(d), 'mean', d.mean(), 'p90', np.percentile(d, 90))
