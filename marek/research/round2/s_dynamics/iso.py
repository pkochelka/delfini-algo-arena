"""Isolated takers: S change across the taker minus c*sign*sqrt(filled): distribution of the impact noise."""
import numpy as np, pandas as pd, sys
import prep
tag = sys.argv[1] if len(sys.argv) > 1 else 'v2'; c = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0278
W = float(sys.argv[3]) if len(sys.argv) > 3 else 0.3
d = prep.build(); r = dict(np.load(prep.common.CACHE / f'smooth_{tag}.npz'))
t = d['t']; m = r['mean']
vol = r['vol'].astype(bool); sh = r['shock'].astype(bool)
tk = np.where(d['ioc'])[0]; tt = t[tk]
iso = np.ones(len(tk), bool); iso[1:] &= np.diff(tt) > W; iso[:-1] &= np.diff(tt) > W
def at(x): return m[np.clip(np.searchsorted(t, x, 'right') - 1, 0, len(t) - 1)]
for nm, reg in [('calm', ~vol & ~sh)]:
    for fcut in ['filled>0', 'filled==0']:
        f = d['filled'][tk]
        sel = iso & reg[tk] & ((f > 0) if fcut == 'filled>0' else (f == 0))
        i = tk[sel]
        dS = at(t[i] + W * 0.9) - at(t[i] - 1e-6)
        sgn = d['sign'][i]; q = np.sqrt(d['filled'][i])
        y = dS - c * sgn * q
        # control: same windows shifted to random calm times without takers
        print(nm, fcut, 'n', len(i), 'mean sign*resid %.4f' % (sgn * y).mean(), 'sd %.4f' % y.std(), 'kurt %.2f' % pd.Series(y).kurt(),
              'pct', np.percentile(y, [1, 5, 25, 50, 75, 95, 99]).round(3))
        if fcut == 'filled>0':
            b = np.polyfit(q, sgn * dS, 1); print('   sign*dS = %.4f*sqrt(f) + %.4f' % tuple(b))
            df = pd.DataFrame(dict(q=q, y2=y ** 2, y=sgn * y))
            print('   resid var / sign-resid by sqrt(filled) quintile:', df.groupby(pd.qcut(df.q, 5)).agg(v=('y2', 'mean'), m=('y', 'mean')).round(4).to_dict())
# control: random times
rng = np.random.default_rng(1); x = rng.uniform(0, 6000, 200000)
j = np.searchsorted(tt, x); ok = (j > 0) & (j < len(tt))
ok[ok] &= (x[ok] - tt[j[ok] - 1] > W) & (tt[j[ok]] - x[ok] > W)
x = x[ok]; x = x[~vol[np.clip(np.searchsorted(t, x) - 1, 0, len(t) - 1)] & ~sh[np.clip(np.searchsorted(t, x) - 1, 0, len(t) - 1)]]
y = at(x + W * 0.9) - at(x)
print('control (no taker in window) n', len(x), 'sd %.4f' % y.std(), 'kurt %.2f' % pd.Series(y).kurt())
