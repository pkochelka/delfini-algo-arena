"""ACF of residual increments dE (E = S - deterministic impact) on a B-s grid, within same-regime runs, shocks removed.
Also distribution of local velocity v = dE over W s windows. Usage: velacf.py TAG params.json [B]"""
import sys, json, numpy as np, pandas as pd
from scipy import stats
import prep, smooth_seg as SS
tag, pf = sys.argv[1], sys.argv[2]; B = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
P = json.load(open(prep.common.HERE / pf))
d = SS.d; r = dict(np.load(prep.common.CACHE / f'smooth_{tag}.npz'))
t = d['t']; m = r['mean']; vol = r['vol'].astype(bool)
def arr(k, dflt): return np.where(vol, P['vol'].get(k, dflt), P['calm'].get(k, dflt)).astype(float)
shift = SS.impact(slice(0, len(t)), arr('c', 0), arr('e', 0.6), arr('pperm', 1.0), arr('tdec', 15.0))
E = m - np.cumsum(np.r_[0, shift[:-1]])
g = np.arange(0, 6000, B); i = np.clip(np.searchsorted(t, g, 'right') - 1, 0, len(t) - 1)
dE = np.diff(E[i]); vg = vol[i[:-1]] & vol[i[1:]]; cg = ~vol[i[:-1]] & ~vol[i[1:]]
SH = pd.read_csv(prep.common.HERE / '../flow/shock_events.csv').onset.to_numpy()
bad = np.zeros(len(dE), bool)
for a in SH: bad[int((a - 0.5) / B):int((a + 6) / B) + 1] = True
VS = pd.read_csv(prep.common.HERE / 'vol_segments.csv')
for a in np.r_[VS.start, VS.end]: bad[max(0, int((a - 1) / B)):int((a + 1) / B) + 1] = True
lags = [1, 2, 3, 4, 6, 8, 10, 14, 20, 30, 40, 60]
for nm, msk in (('calm', cg & ~bad), ('vol', vg & ~bad)):
    x = np.where(msk, dE - dE[msk].mean(), 0.0); v0 = (x[msk] ** 2).mean()
    ac = [(x[:-l] * x[l:])[msk[:-l] & msk[l:]].mean() / v0 for l in lags]
    print(nm, 'B %.2f var/s %.3f  ACF at' % (B, v0 / B), ' '.join('%gs:%.3f' % (l * B, a) for l, a in zip(lags, ac)))
    for W in (1, 2, 4):
        k = int(W / B); c = np.cumsum(np.r_[0, np.where(msk, dE, 0)]); ok = np.r_[0, np.cumsum(~msk)]
        idx = np.arange(0, len(dE) - k, k); good = (ok[idx + k] - ok[idx]) == 0
        v = (c[idx + k] - c[idx])[good] / W
        dv = np.diff(v)[good[1:] & good[:-1]] if False else np.diff(v)
        print('   W %ds velocity sd %.3f kurt %.2f | dv sd %.3f kurt %.2f  corr(v,v_next) %.3f  quantiles |v| 50/90/99: %s' % (
            W, v.std(), stats.kurtosis(v), dv.std(), stats.kurtosis(dv), np.corrcoef(v[1:], v[:-1])[0, 1], np.round(np.quantile(np.abs(v), [.5, .9, .99]), 2)))
