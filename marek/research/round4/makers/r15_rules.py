"""Price and cancel rules of the 15-lot agent (orders from cache/r15_v2.pkl)."""
import numpy as np, pandas as pd
a = pd.read_pickle('cache/r15_v2.pkl')
q = pd.read_pickle('cache/quotes.pkl')
qt = q.t_us.to_numpy(); qb = q.bid.to_numpy(); qa = q.ask.to_numpy()
touch = np.where(a.sg > 0, a.bid, a.ask)
rel = (a.price - touch) * a.sg   # + = better than touch (inside spread)
print('price rel same-side touch at insert, by link:')
print(pd.crosstab(a.link, rel))
opp = np.where(a.sg > 0, a.ask, a.bid)
print('spread at insert', pd.Series(a.ask - a.bid).value_counts().sort_index().head(6).to_dict())
# cancels
c = a[a.end == 'cancelled']
lat = []; never = 0; why = []
for r in c.itertuples():
    i0 = np.searchsorted(qt, r.t_us, 'right'); i1 = np.searchsorted(qt, r.end_t_us, 'left')
    side = qb if r.sg > 0 else qa
    seg = side[i0:i1]
    better = (seg - r.price) * r.sg > 0
    worse = ~np.isnan(seg) & ((seg - r.price) * r.sg < 0)
    if better.any():
        j = np.argmax(better)
        lat.append(r.end_t_us - qt[i0 + j]); why.append('improved')
    else:
        never += 1; lat.append(np.nan); why.append('worse' if worse.any() else 'none')
c = c.assign(lat_us=lat, why=why)
print('cancelled', len(c), c.why.value_counts().to_dict())
l = c.lat_us.dropna() / 1e3
print('latency touch-improve -> cancel row (ms) quantiles', l.quantile([.05, .1, .25, .5, .75, .9, .95]).round(2).to_dict(), 'mean', l.mean().round(2))
print('lifetime (ms) of no-improve cancels', ((c[c.why != 'improved'].end_t_us - c[c.why != 'improved'].t_us) / 1e3).round(1).tolist())
# did the agent cancel EVERY time it was not best? check orders filled/alive that saw an improvement and survived >200ms after
f = a[a.end == 'filled']
surv = []
for r in f.itertuples():
    i0 = np.searchsorted(qt, r.t_us, 'right'); i1 = np.searchsorted(qt, r.end_t_us, 'left')
    side = qb if r.sg > 0 else qa
    better = (side[i0:i1] - r.price) * r.sg > 0
    if better.any():
        surv.append((r.end_t_us - qt[i0 + np.argmax(better)]) / 1e3)
print('filled orders that saw the touch improve first:', len(surv), 'of', len(f), ' time from improve to fill (ms)', np.round(sorted(surv), 1)[:40])
c.to_pickle('cache/r15_cancels.pkl')
# exponential delay fit: mean
from math import log
