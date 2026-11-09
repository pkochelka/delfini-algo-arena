"""Find price jumps on the round-4 tape (non-shock) and how the strategy was positioned going in.
Jump = |mid(t+W) - mid(t)| >= J ticks, greedy non-overlapping, onset refined to the start of the move."""
import sys, numpy as np
g = np.load('vol_signature/cache/grid.npz'); tg = g['g']; M = g['M']; S = g['S']; on = g['on']
x = np.load(sys.argv[1] if len(sys.argv) > 1 else 'jumps/cache/r4_volsig_round4.npz')['x']
lt, lpos = x[:, 0], x[:, 5]
def pos_at(t): return lpos[max(0, np.searchsorted(lt, t) - 1)]
shockmask = np.zeros_like(tg, bool)
for o in on: shockmask |= (tg >= o - 2) & (tg < o + 25)
dt = 0.05
for W, J in [(0.5, 10), (1.0, 15), (1.0, 20), (2.0, 25), (5.0, 40)]:
    k = int(round(W / dt)); d = np.r_[M[k:] - M[:-k], np.zeros(k)]
    ev = []; i = 0
    while i < len(tg) - k:
        if abs(d[i]) >= J and not shockmask[i:i + k].any():
            ev.append(i); i += k
        else: i += 1
    ev = np.array(ev)
    rows = []
    for i in ev:
        sg = np.sign(d[i]); p0 = pos_at(tg[i]); p1 = pos_at(tg[i + k])
        rows.append((sg * d[i], sg * p0, sg * p1, g['vol'][i]))
    r = np.array(rows)
    if not len(r): continue
    mv, a0, a1, vol = r.T
    cap = 100
    print(f'W={W}s J={J}: n={len(r)} (vol {int(vol.sum())}), mean move {mv.mean():.1f}, aligned pos at onset mean {a0.mean():.0f} '
          f'(>=80: {np.mean(a0>=80):.0%}, <=-80: {np.mean(a0<=-80):.0%}, |.|<40: {np.mean(abs(a0)<40):.0%}); after {a1.mean():.0f}; '
          f'captured {np.sum(a0*mv)/1e3:.0f}k of cap {np.sum(cap*mv)/1e3:.0f}k; missed-if-capped {np.sum((cap-a0)*mv)/1e3:.0f}k')
