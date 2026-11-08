"""Are 'ladder-like' singletons (L==1, placed at same-side touch -2..-7, size in the regime's ladder band) the ladder?
Compare their cancel hazard vs touch distance with chain ladders and with other singletons placed at the same levels."""
import numpy as np, pandas as pd
from s_load import with_s
o = with_s()
q = pd.read_pickle('cache/quotes.pkl')
qt = q.t_us.to_numpy(); qb = q.bid.to_numpy().astype(float); qa = q.ask.to_numpy().astype(float)
o['rel'] = np.where(o.sg > 0, o.price - o.bid, o.ask - o.price)
def haz(s, step=5000):
    s = s[s.end == 'cancelled']
    t0 = s.t_us.to_numpy(); t1 = s.cancel_us.to_numpy().astype(np.int64); sg = s.sg.to_numpy(); pr = s.price.to_numpy()
    n = ((t1 - t0)//step).astype(int) + 1
    idx = np.repeat(np.arange(len(s)), n); off = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    tt = t0[idx] + off*step; j = np.searchsorted(qt, tt, 'right') - 1
    rel = (pr[idx] - np.where(sg[idx] > 0, qb[j], qa[j])) * sg[idx]
    jc = np.searchsorted(qt, t1 - 1, 'right') - 1; relc = (pr - np.where(sg > 0, qb[jc], qa[jc])) * sg
    E = np.bincount(np.clip(rel, -9, 0).astype(int) + 9, minlength=10) * step/1e6
    C = np.bincount(np.clip(relc, -9, 0).astype(int) + 9, minlength=10)
    return ' '.join(f'{r}:{c/e:.2f}' for r, c, e in zip(range(-9, 1), C, E) if e > 1)
band = {'calm': (70, 330), 'vol': (12, 75)}
for st_, (lo, hi) in band.items():
    b = o[(o.st == st_) & ~o.ioc]
    dl = (b.rel <= -2) & (b.rel >= -7)
    sets = {'chain ladder': b[b.lad], 'singleton deep, ladder size': b[(b.L == 1) & dl & b.volume.between(lo, hi)],
            'singleton deep, other size': b[(b.L == 1) & dl & ~b.volume.between(lo, hi)]}
    for k, s in sets.items():
        print(f'{st_} {k}: n={len(s)}  life med {np.nanmedian((s.cancel_us - s.t_us)/1e6):.2f} s  hazard by rel: {haz(s.sample(min(len(s), 40000), random_state=0))}')
