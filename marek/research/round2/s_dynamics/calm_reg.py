"""Calm windows: regress dS over window on sum sign*filled^e (and ordered) -> c, exponent, residual var/h."""
import numpy as np, pandas as pd, sys
import prep
tag = sys.argv[1] if len(sys.argv) > 1 else 'v1'
regime = sys.argv[2] if len(sys.argv) > 2 else 'calm'
d = prep.build(); r = dict(np.load(prep.common.CACHE / f'smooth_{tag}.npz'))
t = d['t']; m = r['mean']
R = pd.read_csv(prep.common.ROOT / 'marek/research/round2/flow/regimes_envW.csv')
SH = pd.read_csv(prep.common.ROOT / 'marek/research/round2/flow/shock_events.csv')
g = np.arange(0, 6000, 0.1)
hi = np.zeros(len(g), bool); bad = np.zeros(len(g), bool)
for a, b, hh in zip(R.start, R.end, R.high):
    if hh: hi |= (g >= a) & (g < b)
for a in SH.onset: bad |= (g >= a - 1) & (g < a + 15)
if regime == 'calm':
    bad |= np.convolve(hi, np.ones(41), 'same') > 0
else:
    bad |= ~hi
i = np.clip(np.searchsorted(t, g, 'right') - 1, 0, len(t) - 1)
Sg = m[i]; cb = np.cumsum(bad)
ioc = d['ioc']; tt = t[ioc]; sg = d['sign'][ioc]; f = d['filled'][ioc]; v = d['volume'][ioc]
for hs in [2.0, 5.0]:
    k = int(hs / 0.1)
    a = np.arange(0, len(g) - k, k)
    ok = ((cb[a + k] - cb[a]) == 0) & ~bad[a]
    a = a[ok]
    y = Sg[a + k] - Sg[a]
    ja = np.searchsorted(tt, g[a], 'right'); jb = np.searchsorted(tt, g[a + k], 'right')
    for nm, q in [('filled', f), ('ordered', v)]:
        for e in [0.4, 0.5, 0.6]:
            cs = np.concatenate([[0], np.cumsum(sg * q ** e)])
            x = cs[jb] - cs[ja]
            c = (x * y).sum() / (x * x).sum(); res = y - c * x
            print(regime, 'h', hs, nm, 'e', e, 'n', len(y), 'c %.4f' % c, 'R2 %.3f' % (1 - res.var() / y.var()), 'resid var/h %.4f' % (res.var() / hs), 'kurt %.2f' % pd.Series(res).kurt())
# residual dependence on activity (h=2, e=0.5)
k = 20
a = np.arange(0, len(g) - k, k); ok = ((cb[a + k] - cb[a]) == 0) & ~bad[a]; a = a[ok]
y = Sg[a + k] - Sg[a]
ja = np.searchsorted(tt, g[a], 'right'); jb = np.searchsorted(tt, g[a + k], 'right')
cs = np.concatenate([[0], np.cumsum(sg * f ** 0.5)]); x = cs[jb] - cs[ja]
c = (x * y).sum() / (x * x).sum(); res = y - c * x
ntk = jb - ja
cf = np.concatenate([[0], np.cumsum(f)]); sf = cf[jb] - cf[ja]
df = pd.DataFrame(dict(res2=res ** 2, ntk=ntk, sf=sf))
print(df.groupby(pd.qcut(df.ntk, 5, duplicates='drop')).res2.agg(['mean', 'size']).round(3))
print(df.groupby(pd.qcut(df.sf, 5)).res2.agg(['mean', 'size']).round(3))
