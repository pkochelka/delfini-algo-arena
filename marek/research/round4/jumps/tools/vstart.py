"""Capture around volatile-episode starts: sum pos*dmid (carry) vs 100*|dmid| over 50 ms steps, by time since vol start."""
import numpy as np, csv
z = np.load('jumps/cache/book_r4_volsig.npz'); x = z['x']
t, F, sd, kv, Y, turb, pos, bb, ba = x.T
mid = (bb + ba) / 2
g = np.load('vol_signature/cache/grid.npz'); on = g['on']
shock = np.zeros(len(t), bool)
for o in on: shock |= (t >= o - 2) & (t < o + 25)
reg = [(float(r['start']), float(r['end']), r['regime']) for r in csv.DictReader(open('flow/regimes_1s.csv'))]
vs = np.array([a for a, b, r in reg if r == 'volatile'])
since = np.full(len(t), 1e9)
for a in vs:
    m = (t >= a - 3) & (t < a + 60); since[m] = np.minimum(np.where(since[m] > 1e8, t[m] - a, since[m]), t[m] - a)
dm = np.r_[np.diff(mid), 0]; carry = pos * dm
for lo, hi in [(-3, -1), (-1, 0), (0, 1), (1, 2), (2, 5), (5, 10), (10, 60)]:
    m = (since >= lo) & (since < hi) & ~shock
    print(f'since vol start [{lo},{hi}) s: n={m.sum():6d} carry {carry[m].sum()/1e3:7.1f}k  cap-ceiling {100*abs(dm[m]).sum()/1e3:7.1f}k  '
          f'ratio {carry[m].sum()/(100*abs(dm[m]).sum()+1e-9):.2f}  turb {turb[m].mean():.2f}  |kv| {abs(kv[m]).mean():.2f}')
m = (turb > 0) & ~shock & (since > 60); print('other turb ratio', carry[m].sum() / (100 * abs(dm[m]).sum()))
m = (turb == 0) & ~shock & (since > 60); print('calm (far from vol) ratio', carry[m].sum() / (100 * abs(dm[m]).sum()), carry[m].sum()/1e3)
