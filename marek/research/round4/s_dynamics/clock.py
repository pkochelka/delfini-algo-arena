"""Does the velocity change on a clock? Residual increments dE on a fine grid (B s, aligned to integer seconds):
correlation of pairs at the same lag, both inside one period vs straddling a period boundary, for period PER and phase.
Usage: clock.py TAG params.json [B] [PER]"""
import sys, json, numpy as np, pandas as pd
import prep, smooth_seg as SS
tag, pf = sys.argv[1], sys.argv[2]; B = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1
P = json.load(open(prep.common.HERE / pf))
d = SS.d; r = dict(np.load(prep.common.CACHE / f'smooth_{tag}.npz'))
t = d['t']; m = r['mean']; vol = r['vol'].astype(bool)
def arr(k, dflt): return np.where(vol, P['vol'].get(k, dflt), P['calm'].get(k, dflt)).astype(float)
d['vshift'][:] = 0.0
shift = SS.impact(slice(0, len(t)), arr('c', 0), arr('e', 0.6), arr('pperm', 1.0), arr('tdec', 15.0))
E = m - np.cumsum(np.r_[0, shift[:-1]])
g = np.arange(0, 6000, B); i = np.clip(np.searchsorted(t, g, 'right') - 1, 0, len(t) - 1)
y = np.diff(E[i]); vg = vol[i[1:]] & vol[i[:-1]]
SH = pd.read_csv(prep.common.HERE / '../flow/shock_events.csv').onset.to_numpy()
bad = np.zeros(len(y), bool)
for a in SH: bad[int((a - 0.5) / B):int((a + 6) / B) + 1] = True
nb = int(round(1 / B))
for per in [float(x) for x in (sys.argv[4].split(',') if len(sys.argv) > 4 else ['1.0'])]:
    npb = int(round(per / B))
    for nm, msk in (('vol', vg & ~bad), ('calm', ~vg & ~bad)):
        x = np.where(msk, y - y[msk].mean(), 0); v0 = (x[msk] ** 2).mean()
        out = []
        for lag in (1, 2, 3):
            for ph in range(npb):
                pass
            k = np.arange(len(y) - lag)
            ok = msk[k] & msk[k + lag]
            same = (k // npb) == ((k + lag) // npb)
            a = (x[k] * x[k + lag])[ok & same].mean() / v0; b = (x[k] * x[k + lag])[ok & ~same].mean() / v0
            out.append('lag %.1fs same %.3f cross %.3f' % (lag * B, a, b))
        print('period %.2f %s: %s' % (per, nm, ' | '.join(out)))
# phase profile of |dE| (vol) and of squared change in local slope
x = y.copy(); msk = vg & ~bad
ph = (np.arange(len(y)) % nb)
dd = np.abs(np.diff(y))       # change in increment (proxy for velocity change), at boundary between bins k,k+1
php = (np.arange(1, len(y)) % nb)
ok = msk[1:] & msk[:-1]
print('vol |dE| by phase:', np.round([np.abs(y[msk & (ph == p)]).mean() for p in range(nb)], 3))
print('vol |d(dE)| at boundary phase:', np.round([dd[ok & (php == p)].mean() for p in range(nb)], 3))
