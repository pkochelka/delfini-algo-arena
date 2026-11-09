"""Turbulent, non-shock: future mid move vs the filter's state (F - mid, kv) in bins: is the response convex in |kv|?"""
import numpy as np
z = np.load('jumps/cache/book_r4_volsig.npz'); x = z['x']
t, F, sd, kv, Y, turb, pos, bb, ba = x.T
g = np.load('vol_signature/cache/grid.npz'); on = g['on']
shock = np.zeros(len(t), bool)
for o in on: shock |= (t >= o - 2) & (t < o + 25)
mid = (bb + ba) / 2; dt = np.median(np.diff(t))
F0 = F - 0.15 * kv * (turb > 0)  # volsig logs the leaned F? check below
sel = (turb > 0) & ~shock
for h in (0.25, 0.5, 1.0, 2.0):
    k = int(round(h / dt)); fut = np.r_[mid[k:] - mid[:-k], np.full(k, np.nan)]
    s = sel & np.isfinite(fut)
    gap = F[s] - mid[s]; v = kv[s]; y = fut[s]
    A = np.c_[gap, v, np.ones_like(v)]; b, *_ = np.linalg.lstsq(A, y, rcond=None)
    res = y - A @ b
    print(f'h={h}: beta gap {b[0]:.2f} kv {b[1]:.2f}; residual by |kv| bin (signed by kv):')
    edges = [0, 1, 2, 3, 4, 6, 8, 12, 99]
    line = ''
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (abs(v) >= lo) & (abs(v) < hi)
        if m.sum() > 50: line += f'  [{lo},{hi}) n={m.sum()} r={np.mean(res[m]*np.sign(v[m])):+.2f} pos·sgn={np.mean(pos[s][m]*np.sign(v[m])):+.0f}'
    print(line)
    edges = [0, 2, 4, 6, 8, 12, 99]; line = ''
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (abs(gap) >= lo) & (abs(gap) < hi)
        if m.sum() > 50: line += f'  gap[{lo},{hi}) n={m.sum()} r={np.mean(res[m]*np.sign(gap[m])):+.2f}'
    print(line)
print('kv sd turb', kv[sel].std(), 'gap sd', (F - mid)[sel].std())
