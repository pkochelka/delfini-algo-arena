"""Hazard of ladder-order cancel rows vs current distance from the same-side touch (rel = ticks, <0 behind touch)
and vs age. Exposure sampled every 5 ms; events at the cancel row. Compares with non-ladder (L==1) sizes 15-80 (vol)."""
import numpy as np, pandas as pd
o = pd.read_pickle('cache/chains.pkl')
g = np.load('cache/regime_sz.npz')['post']
o['vol'] = g[(o.t_us//1_000_000).to_numpy(), 1] > .5
q = pd.read_pickle('cache/quotes.pkl')
qt = q.t_us.to_numpy(); qb = q.bid.to_numpy().astype(float); qa = q.ask.to_numpy().astype(float)
def touch(t, sg):
    j = np.searchsorted(qt, t, 'right') - 1
    return np.where(sg > 0, qb[j], qa[j])
def run(sel, name, step=5000):
    s = sel[sel.end == 'cancelled']
    t0 = s.t_us.to_numpy(); t1 = s.cancel_us.to_numpy(); sg = s.sg.to_numpy(); pr = s.price.to_numpy()
    n = ((t1 - t0)//step).astype(int) + 1
    idx = np.repeat(np.arange(len(s)), n); off = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    tt = t0[idx] + off*step
    rel = (pr[idx] - touch(tt, sg[idx])) * sg[idx]
    relc = (pr - touch(t1 - 1, sg)) * sg      # just before cancel row
    rel_bins = np.arange(-12, 3)
    E = np.bincount(np.clip(rel, -12, 2).astype(int) + 12, minlength=15) * step/1e6
    C = np.bincount(np.clip(relc, -12, 2).astype(int) + 12, minlength=15)
    print(name, 'n', len(s))
    for r_, e_, c_ in zip(rel_bins, E, C):
        if e_ > 1: print(f'   rel {r_:3d}: exposure {e_:9.1f} s  cancels {c_:7d}  hazard {c_/e_:6.2f} /s')
    age = (tt - t0[idx])/1e6; agec = (t1 - t0)/1e6
    ab = np.array([0, .01, .025, .05, .1, .2, .5, 1, 2, 5, 10, 20, 100])
    Ea = np.histogram(age, ab)[0]*step/1e6; Ca = np.histogram(agec, ab)[0]
    print('   age hazard:', ' '.join(f'{a}-{b}:{c/e:.2f}' for a, b, c, e in zip(ab[:-1], ab[1:], Ca, Ea) if e > 0))
lad = o[(o.L >= 2) & (o.L <= 6) & ~o.ioc.astype(bool)]
run(lad[lad.vol], 'ladder vol'); run(lad[~lad.vol], 'ladder calm')
base = o[(o.L == 1) & ~o.ioc.astype(bool)]
run(base[base.vol & (base.volume < 10)].sample(60000, random_state=0), 'base vol small (<10)')
run(base[~base.vol & (base.volume < 10)].sample(60000, random_state=0), 'base calm small (<10)')
