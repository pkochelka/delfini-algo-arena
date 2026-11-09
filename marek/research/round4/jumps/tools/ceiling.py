"""Perfect-foresight ceiling of non-shock burst oversize: at every turbulent 50 ms point, the best direction's
E-lot round trip held tau s (sweep in, sweep out), net of 10k*tau; greedy pick of the best non-overlapping windows."""
import numpy as np
z = np.load('jumps/cache/book_r4_volsig.npz'); x, db, da = z['x'], z['db'], z['da']
t, F, sd, kv, Y, turb, pos, bb, ba = x.T
g = np.load('vol_signature/cache/grid.npz'); on = g['on']
shock = np.zeros(len(t), bool)
for o in on: shock |= (t >= o - 2) & (t < o + 25)
def sw(D, E):
    c = np.cumsum(D, 1); k = (c < E).sum(1); k = np.minimum(k, 19)
    prev = np.where(k > 0, np.take_along_axis(c, np.maximum(k - 1, 0)[:, None], 1)[:, 0], 0)
    lv = np.arange(20)[None, :]
    inner = (D * lv * (lv < k[:, None])).sum(1)
    out = (inner + (E - prev) * k) / E
    out[c[:, -1] < E] = np.nan
    return out
for E in (300, 900):
    sa, sb = sw(da, E), sw(db, E)
    for tau in (0.5, 1.0, 2.0, 4.0):
        k = int(round(tau / 0.05)); n = len(t) - k
        up = (bb[k:] - sb[k:]) - (ba[:n] + sa[:n]); dn = (bb[:n] - sb[:n]) - (ba[k:] + sa[k:])
        best = np.fmax(up, dn) * E - 10000 * tau
        ok = (turb[:n] > 0) & ~shock[:n] & ~shock[k:k + n] & np.isfinite(best)
        val = np.where(ok, best, -np.inf)
        order = np.argsort(-val); used = np.zeros(len(t), bool); tot = 0; cnt = 0
        for i in order:
            if val[i] <= 0: break
            if used[i:i + k + 1].any(): continue
            used[i:i + k + 1] = True; tot += val[i]; cnt += 1
        print(f'E={E} tau={tau}: perfect-foresight windows {cnt}, ceiling {tot/1e3:.0f}k  (positive-net share of turb points {np.mean(val[ok] > 0):.3%})')
