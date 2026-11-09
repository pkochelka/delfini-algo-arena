"""Burst-oversize economics: at turbulent moments with a strong directional signal, take E extra lots by sweeping the
book, hold tau s, sweep back out. net = E*(exit avg - entry avg) - 10k*tau. Non-overlapping, greedy in time."""
import numpy as np
z = np.load('jumps/cache/book_r4_volsig.npz'); x, db, da = z['x'], z['db'], z['da']
t, F, sd, kv, Y, turb, pos, bb, ba = x.T
g = np.load('vol_signature/cache/grid.npz'); on = g['on']
shock = np.zeros(len(t), bool)
for o in on: shock |= (t >= o - 2) & (t < o + 25)
mid = (bb + ba) / 2
def sweep(depth, E):  # avg ticks beyond touch to fill E lots; inf if not enough depth in 20 levels
    c = np.cumsum(depth); 
    if c[-1] < E: return np.nan
    k = np.searchsorted(c, E); prev = c[k - 1] if k else 0
    return ((depth[:k] * np.arange(k)).sum() + (E - prev) * k) / E
dt = np.median(np.diff(t))
def run(sig, thr, E, tau, sel=None):
    n = len(t); k = int(round(tau / dt)); i = 0; out = []
    while i < n - k - 1:
        if turb[i] and not shock[i] and abs(sig[i]) >= thr and (sel is None or sel[i]):
            d = np.sign(sig[i])
            if d > 0:
                e_in = ba[i] + sweep(da[i], E); e_out = bb[i + k] - sweep(db[i + k], E)
                pnl = E * (e_out - e_in)
            else:
                e_in = bb[i] - sweep(db[i], E); e_out = ba[i + k] + sweep(da[i + k], E)
                pnl = E * (e_in - e_out)
            if np.isfinite(pnl):
                out.append((t[i], pnl - 10000 * tau, pnl / E, d * (mid[i + k] - mid[i]))); i += k + 1; continue
        i += 1
    return np.array(out)
print('turb frac', turb.mean(), 'median depth ask lvls0-4', np.median(da[turb > 0][:, :5], 0), 'cum20', np.median(da[turb > 0].sum(1)))
fh = t < 3000
for name, sig in [('kv', kv), ('F-mid', F - mid), ('kv+F-mid', 0.5 * kv + (F - mid))]:
    for E in (200, 500, 900):
        for tau in (0.5, 1.0, 2.0):
            best = None
            for thr in np.percentile(abs(sig[turb > 0]), [90, 95, 98, 99, 99.5]):
                r = run(sig, thr, E, tau)
                if not len(r): continue
                h1 = r[r[:, 0] < 3000, 1].sum(); h2 = r[r[:, 0] >= 3000, 1].sum()
                line = f'{name:9s} E={E} tau={tau} thr={thr:5.1f} n={len(r):4d} net={r[:,1].sum()/1e3:7.0f}k (h1 {h1/1e3:6.0f} h2 {h2/1e3:6.0f}) gross/lot {r[:,2].mean():5.2f} mid move {r[:,3].mean():5.2f}'
                if best is None or r[:, 1].sum() > best[0]: best = (r[:, 1].sum(), line)
            print(best[1])
