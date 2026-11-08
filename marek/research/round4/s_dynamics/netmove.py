"""Where does the net S move come from? Sum of 1 s increments of S, impact I, residual E by regime; shocks separately."""
import sys, json, numpy as np, pandas as pd
import prep, smooth_seg as SS
tag, pf = sys.argv[1], sys.argv[2]
P = json.load(open(prep.common.HERE / pf))
d = SS.d; r = dict(np.load(prep.common.CACHE / f'smooth_{tag}.npz'))
t = d['t']; m = r['mean']; vol = r['vol'].astype(bool)
def arr(k, dflt): return np.where(vol, P['vol'].get(k, dflt), P['calm'].get(k, dflt)).astype(float)
d['vshift'][:] = 0.0
shift = SS.impact(slice(0, len(t)), arr('c', 0), arr('e', 0.6), arr('pperm', 1.0), arr('tdec', 15.0))
I = np.cumsum(np.r_[0, shift[:-1]]); E = m - I
g = np.arange(0, 6001, 1.0); i = np.clip(np.searchsorted(t, g, 'right') - 1, 0, len(t) - 1)
yS, yI, yE = np.diff(m[i]), np.diff(I[i]), np.diff(E[i]); vg = vol[i[:-1]]
SH = pd.read_csv(prep.common.HERE / '../flow/shock_events.csv').onset.to_numpy()
bad = np.zeros(len(yS), bool)
for a in SH: bad[int(a - 0.5):int(a + 6) + 1] = True
print('S start %.1f end %.1f min %.1f max %.1f' % (m[0], m[-1], m.min(), m.max()))
for nm, w in (('calm', ~vg & ~bad), ('vol', vg & ~bad), ('shock', bad)):
    print('%-5s secs %4d  net S %8.1f  net impact %8.1f  net E %8.1f   E per s %.3f (se %.3f)' % (nm, w.sum(), yS[w].sum(), yI[w].sum(), yE[w].sum(), yE[w].mean(), yE[w].std() / np.sqrt(w.sum()) * np.sqrt(5)))
# buys vs sells: mean sign*J at fills by side in vol, from smoothed path (J = S after - S before)
n = len(t); ti = np.flatnonzero(d['ioc'] & (d['filled'] > 0) & (np.arange(n) < n - 1))
J = m[ti + 1] - m[ti]
for nm, w in (('calm', ~vol[ti]), ('vol', vol[ti])):
    for s in (1, -1):
        ww = w & (d['sign'][ti] == s)
        k = arr('c', 0)[ti][ww] * d['filled'][ti][ww] ** arr('e', .6)[ti][ww]
        print(nm, 'side', s, 'n', ww.sum(), 'mean |J| along side %.3f model %.3f' % ((s * J[ww]).mean(), k.mean()))
