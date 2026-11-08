"""Exact cancel delay: ladder orders whose same-side touch moves so the order is > 7 ticks behind (rel <= -8). The touch
change has an exact timestamp; the cancel row follows. Delay histogram (only orders that stay at <= -8 until the row)."""
import numpy as np, pandas as pd
from s_load import with_s
o = with_s()
q = pd.read_pickle('cache/quotes.pkl'); qt = q.t_us.to_numpy(); qb = q.bid.to_numpy().astype(float); qa = q.ask.to_numpy().astype(float)
lad = o[o.lad & (o.end == 'cancelled')]
delays = []; back = 0
for r in lad.itertuples():
    j0 = np.searchsorted(qt, r.t_us, 'right'); j1 = np.searchsorted(qt, r.cancel_us, 'right')
    tb = (qb if r.sg > 0 else qa)[j0 - 1:j1]
    rel = (r.price - tb) * r.sg
    far = np.flatnonzero(rel <= -8)
    if not len(far): continue
    k = far[0]
    if (rel[k:] > -8).any(): back += 1; continue      # moved back before the row
    t_ev = qt[j0 - 1 + k] if k > 0 else r.t_us
    delays.append((r.cancel_us - t_ev) / 1e3)
d = np.array(delays)
print('n', len(d), 'moved back (excluded)', back)
print('quantiles ms', np.percentile(d, [1, 5, 10, 25, 50, 75, 90, 95, 99]).round(2), 'mean', d.mean().round(2))
h, e = np.histogram(d, np.r_[0:10:1, 10:30:2, 30:60:5, 60:121:20])
for a, b, c in zip(e[:-1], e[1:], h): print(f'{a:5.0f}-{b:5.0f} ms {c:6d} density {c/(b-a)/len(d)*1000:7.2f} /s')
x = d[d > 6] - 6
print('exp fit beyond 6 ms: rate %.1f /s; share < 5 ms %.4f' % (1000 / x.mean(), (d < 5).mean()))
np.save('cache/ladder_far_delays.npy', d)
