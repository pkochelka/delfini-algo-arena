"""Calm-regime residual diffusion of X = S - c*sum sign sqrt(filled): var/h across horizons, windows fully calm."""
import numpy as np, pandas as pd, sys
import prep
tag = sys.argv[1] if len(sys.argv) > 1 else 'v1'
c = float(sys.argv[2]) if len(sys.argv) > 2 else 0.026
d = prep.build(); r = dict(np.load(prep.common.CACHE / f'smooth_{tag}.npz'))
t = d['t']; m = r['mean']
imp = np.where(d['ioc'], c * d['sign'] * np.sqrt(d['filled']), 0)
X = m - (np.cumsum(imp) - imp)
R = pd.read_csv(prep.common.ROOT / 'marek/research/round2/flow/regimes_envW.csv')
SH = pd.read_csv(prep.common.ROOT / 'marek/research/round2/flow/shock_events.csv')
g = np.arange(0, 6000, 0.1)
bad = np.zeros(len(g), bool)
for a, b, hh in zip(R.start, R.end, R.high):
    if hh: bad |= (g >= a - 2) & (g < b + 2)
for a in SH.onset: bad |= (g >= a - 1) & (g < a + 15)
i = np.clip(np.searchsorted(t, g, 'right') - 1, 0, len(t) - 1)
Xg = X[i]; sdg = r['sd'][i]
cb = np.cumsum(bad)
for hs in [0.5, 1, 2, 5, 10, 30, 60, 120]:
    k = int(round(hs / 0.1))
    a = np.arange(0, len(g) - k, max(1, k // 2))
    ok = (cb[a + k] - cb[a]) == 0
    ok &= ~bad[a]
    dd = Xg[a[ok] + k] - Xg[a[ok]]
    noise = (sdg[a[ok]] ** 2 + sdg[a[ok] + k] ** 2).mean()
    print('h %6.1f n %6d var/h %.4f  (minus post-sd noise %.4f) kurt %.2f' % (hs, ok.sum(), dd.var() / hs, (dd.var() - noise) / hs, pd.Series(dd).kurt()))
# drought test
for a in []: pass
v_d = []; v_n = []
for a in SH.onset:
    for (lo, hi_, lst) in [(a + 0.3, a + 2.8, v_d), (a - 30, a - 27.5, v_n)]:
        j0 = np.searchsorted(t, lo); j1 = np.searchsorted(t, hi_)
        lst.append(X[j1] - X[j0])
v_d = np.array(v_d); v_n = np.array(v_n)
print('drought 2.5s: var/h %.3f ; pre-shock control var/h %.3f  (n=%d)' % ((v_d ** 2).mean() / 2.5, (v_n ** 2).mean() / 2.5, len(v_d)))
