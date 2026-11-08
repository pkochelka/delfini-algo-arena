"""Ladder orders near the touch (rel >= -1): hazard vs time since entering the near zone; and fate (fill vs cancel)."""
import numpy as np, pandas as pd
o = pd.read_pickle('cache/chains.pkl')
g = np.load('cache/regime_sz.npz')['post']
o['vol'] = g[(o.t_us//1_000_000).to_numpy(), 1] > .5
q = pd.read_pickle('cache/quotes.pkl')
qt = q.t_us.to_numpy(); qb = q.bid.to_numpy().astype(float); qa = q.ask.to_numpy().astype(float)
lad = o[(o.L >= 2) & (o.L <= 6) & ~o.ioc.astype(bool)]
step = 2000
for v in (False, True):
    s = lad[lad.vol == v]
    t0 = s.t_us.to_numpy(); t1 = s.end_t_us.fillna(6e9).to_numpy().astype(np.int64); sg = s.sg.to_numpy(); pr = s.price.to_numpy()
    canc = (s.end == 'cancelled').to_numpy(); filled = (s.end == 'filled').to_numpy()
    n = ((t1 - t0)//step).astype(int) + 1
    idx = np.repeat(np.arange(len(s)), n); off = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    tt = t0[idx] + off*step
    j = np.searchsorted(qt, tt, 'right') - 1
    rel = (pr[idx] - np.where(sg[idx] > 0, qb[j], qa[j])) * sg[idx]
    near = rel >= -1
    # first time near per order
    first = np.full(len(s), -1, np.int64)
    ni = idx[near]; nt = tt[near]
    u, fi = np.unique(ni, return_index=True); first[u] = nt[fi]
    ever = first >= 0
    print('VOL' if v else 'CALM', 'orders', len(s), 'ever near', ever.mean().round(3),
          'fate of ever-near: filled', filled[ever].mean().round(3), 'cancelled', canc[ever].mean().round(3),
          '| never near: filled', filled[~ever].mean().round(3))
    # time-since-near hazard for ever-near orders (exposure from first near to end)
    tsn = (tt - first[idx])/1e3
    m = ever[idx] & (tsn >= 0)
    bins = np.array([0, 10, 25, 50, 100, 200, 500, 1000, 3000, 1e9])
    E = np.histogram(tsn[m], bins)[0]*step/1e6
    ce = canc & ever; C = np.histogram((t1[ce] - first[ce])/1e3, bins)[0]
    F = np.histogram((t1[filled & ever] - first[filled & ever])/1e3, bins)[0]
    print('  hazard since first near (ms): cancel', ' '.join(f'{int(a)}-:{c/e:.2f}' for a, c, e in zip(bins[:-1], C, E)))
    print('                               fill  ', ' '.join(f'{int(a)}-:{c/e:.2f}' for a, c, e in zip(bins[:-1], F, E)))
