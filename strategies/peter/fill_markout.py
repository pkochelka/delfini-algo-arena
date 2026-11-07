"""Markouts of the fills recorded by fill_probe.py: overall, by share left on our side, queue position, age.

uv run python strategies/peter/fill_markout.py logs/probe/wf09.npz
"""
import sys, numpy as np
d = np.load(sys.argv[1]); m, f = d['mids'], d['fills']
t, sg, q, px, qb, qa, bb, ba, pos, ahead, tot, age = f.T
def mid_at(tt):
    i = np.searchsorted(m[:, 0], tt, 'right') - 1; return m[np.maximum(i, 0), 1]
own_share = np.where(sg > 0, qb / (qb + qa), qa / (qb + qa))  # market share of OUR side at fill (after the hit)
print(f'fills {len(f)}  lots {q.sum():.0f}')
for H in (0, 0.1, 1, 5, 30):
    mo = sg * (mid_at(t + H * 1e6) - px)  # maker PnL per lot vs fill price
    print(f'markout +{H:>4}s: {np.average(mo, weights=q):+.3f} ticks/lot  total {np.sum(mo * q):+.0f}')
mo1 = sg * (mid_at(t + 1e6) - px)
print('\nby our-side share of touch (market qty) at the fill:')
for lo, hi in [(0, .05), (.05, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.01)]:
    k = (own_share >= lo) & (own_share < hi)
    if k.sum(): print(f'  {lo:.2f}-{hi:.2f}: lots {q[k].sum():7.0f} ({q[k].sum()/q.sum():.2f})  markout1s {np.average(mo1[k], weights=q[k]):+.3f}  total {np.sum(mo1[k]*q[k]):+.0f}')
fill_dist = np.where(sg > 0, bb - px, px - ba)  # >0 = filled behind the current touch (book moved through)
print('\nby fill price vs post-fill touch (0 = at touch, <0 = level got eaten):')
for v in np.unique(fill_dist):
    k = fill_dist == v; print(f'  {v:+.0f}: lots {q[k].sum():7.0f} markout1s {np.average(mo1[k], weights=q[k]):+.3f}')
adds = sg * pos > 0
print(f'\nfills that add to |pos| after: lots {q[adds].sum():.0f}  mo1 {np.average(mo1[adds], weights=q[adds]):+.3f}; reducing: {q[~adds].sum():.0f} mo1 {np.average(mo1[~adds], weights=q[~adds]):+.3f}')

print('by market qty ahead of our order (last snapshot before fill):')
for lo, hi in [(-1,0),(0,1),(1,100),(100,300),(300,700),(700,1500),(1500,1e9)]:
    k=(ahead>=lo)&(ahead<hi)
    if k.sum(): print(f'  [{lo},{hi}): lots {q[k].sum():7.0f} ({q[k].sum()/q.sum():.2f}) markout1s {np.average(mo1[k],weights=q[k]):+.3f} mo0.1 {np.average((sg*(mid_at(t+1e5)-px))[k],weights=q[k]):+.3f}')
print('by order age at fill (s):')
for lo,hi in [(0,.1),(.1,.5),(.5,2),(2,10),(10,1e9)]:
    k=(age/1e6>=lo)&(age/1e6<hi)
    if k.sum(): print(f'  [{lo},{hi}): lots {q[k].sum():7.0f} markout1s {np.average(mo1[k],weights=q[k]):+.3f}')
