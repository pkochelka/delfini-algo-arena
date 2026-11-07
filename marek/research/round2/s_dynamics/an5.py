import numpy as np, pandas as pd, sys
import prep
d = prep.build(); tag = sys.argv[1] if len(sys.argv) > 1 else 'v1'
r = dict(np.load(prep.common.CACHE / f'smooth_{tag}.npz'))
t = d['t']; m = r['mean']
imp = np.where(d['ioc'], 0.026 * d['sign'] * np.sqrt(d['filled']), 0)
cumimp = np.cumsum(imp) - imp  # impact strictly before event i
X = m - cumimp
def at(arr, tt):
    i = np.clip(np.searchsorted(t, tt, 'right') - 1, 0, len(t) - 1); return arr[i]
g = np.arange(0, 6000, 0.05)
Xg = at(X, g); Sg = at(m, g)
for name, A in [('S', Sg), ('S - impact', Xg)]:
    print(name)
    for hs in [0.05, 0.1, 0.5, 1, 5, 10, 60, 300]:
        k = int(round(hs / 0.05)); dd = A[k::k] - A[:-k:k]
        print('  h=%6.2f var/h %.3f kurt %.1f' % (hs, dd.var() / hs, pd.Series(dd).kurt()))
# per 100s bin realized variance of 1s increments of X
k = 20; dd = Xg[k::k] - Xg[:-k:k]; tt = g[:-k:k]
df = pd.DataFrame(dict(t=tt, d=dd))
b = df.groupby((df.t // 100).astype(int)).d.agg(lambda z: (z ** 2).mean())
print('1s RV per 100s bin:'); print(' '.join('%.1f' % v for v in b.to_numpy()))
ac = [np.corrcoef(dd[:-L], dd[L:])[0, 1] for L in range(1, 11)]
print('ACF 1s incr', np.round(ac, 3))
ad = np.abs(dd); print('ACF |1s incr|', np.round([np.corrcoef(ad[:-L], ad[L:])[0, 1] for L in [1, 2, 5, 10, 30, 60, 120, 300, 600]], 3))
np.save(prep.common.CACHE / f'Xg_{tag}.npy', np.stack([g, Xg, Sg]))
