"""Ladder cancel hazard vs distance from S (prelim S) jointly with touch distance. S(t) = S at the latest insert."""
import sys, numpy as np, pandas as pd
tag = sys.argv[1] if len(sys.argv) > 1 else 'smooth_v2r2'
z = np.load(f'../s_dynamics/_cache/{tag}.npz')
o = pd.read_pickle('cache/chains.pkl')
St = o.t_us.to_numpy(); Sv = z['mean']
g = np.load('cache/regime_sz.npz')['post']
o['vol'] = g[(o.t_us//1_000_000).to_numpy(), 1] > .5
q = pd.read_pickle('cache/quotes.pkl')
qt = q.t_us.to_numpy(); qb = q.bid.to_numpy().astype(float); qa = q.ask.to_numpy().astype(float)
def at(tarr, varr, t): return varr[np.searchsorted(tarr, t, 'right') - 1]
lad = o[(o.L >= 2) & (o.L <= 6) & ~o.ioc.astype(bool) & (o.end == 'cancelled')]
step = 5000; LAG = 25000   # look at state 25 ms before the cancel row (decision time)
for v in (False, True):
    s = lad[lad.vol == v]
    t0 = s.t_us.to_numpy(); t1 = s.cancel_us.to_numpy(); sg = s.sg.to_numpy(); pr = s.price.to_numpy()
    n = ((t1 - t0)//step).astype(int) + 1
    idx = np.repeat(np.arange(len(s)), n); off = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    tt = t0[idx] + off*step
    dS = (at(St, Sv, tt) - pr[idx]) * sg[idx]          # >0: order behind S
    rel = (pr[idx] - np.where(sg[idx] > 0, at(qt, qb, tt), at(qt, qa, tt))) * sg[idx]
    tc = np.maximum(t1 - LAG, t0)
    dSc = (at(St, Sv, tc) - pr) * sg; relc = (pr - np.where(sg > 0, at(qt, qb, tc), at(qt, qa, tc))) * sg
    bins = np.array([-5, 0, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9, 20])
    print('VOL' if v else 'CALM', 'hazard by dS (S - price, ticks behind S), only rel in [-7,-2] (touch window):')
    w = (rel >= -7) & (rel <= -2); wc = (relc >= -7) & (relc <= -2)
    E = np.histogram(dS[w], bins)[0]*step/1e6; C = np.histogram(dSc[wc], bins)[0]
    print('   ', ' '.join(f'[{a},{b}):{c/e:.2f}({c})' for a, b, c, e in zip(bins[:-1], bins[1:], C, E) if e > 0.5))
    print('  rel in {-1,0}:')
    w = rel >= -1; wc = relc >= -1
    E = np.histogram(dS[w], bins)[0]*step/1e6; C = np.histogram(dSc[wc], bins)[0]
    print('   ', ' '.join(f'[{a},{b}):{c/e:.2f}({c})' for a, b, c, e in zip(bins[:-1], bins[1:], C, E) if e > 0.5))
