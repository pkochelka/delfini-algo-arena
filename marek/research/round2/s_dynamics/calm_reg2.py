"""Window regressions dS ~ a*sum(sign) + c*sum(sign*sqrt(filled)) [+ b*sum(sign*1[filled==0])], per regime."""
import numpy as np, pandas as pd, sys
import prep
tag = sys.argv[1] if len(sys.argv) > 1 else 'v1'
regime = sys.argv[2] if len(sys.argv) > 2 else 'calm'
hs = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
d = prep.build(); r = dict(np.load(prep.common.CACHE / f'smooth_{tag}.npz'))
t = d['t']; m = r['mean']
R = pd.read_csv(prep.common.ROOT / 'marek/research/round2/flow/regimes_envW.csv')
SH = pd.read_csv(prep.common.ROOT / 'marek/research/round2/flow/shock_events.csv')
g = np.arange(0, 6000, 0.1)
hi = np.zeros(len(g), bool); bad = np.zeros(len(g), bool)
for a, b, hh in zip(R.start, R.end, R.high):
    if hh and b - a > 3.5: hi |= (g >= a) & (g < b)
for a in SH.onset: bad |= (g >= a - 1) & (g < a + 15)
if regime == 'calm':
    bad |= np.convolve(hi, np.ones(41), 'same') > 0
else:
    bad |= ~hi
i = np.clip(np.searchsorted(t, g, 'right') - 1, 0, len(t) - 1)
Sg = m[i]; cb = np.cumsum(bad)
ioc = d['ioc']; tt = t[ioc]; sg = d['sign'][ioc]; f = d['filled'][ioc]; v = d['volume'][ioc]
k = int(hs / 0.1)
a = np.arange(0, len(g) - k, k)
ok = ((cb[a + k] - cb[a]) == 0) & ~bad[a]; a = a[ok]
y = Sg[a + k] - Sg[a]
ja = np.searchsorted(tt, g[a], 'right'); jb = np.searchsorted(tt, g[a + k], 'right')
def S_(z):
    cs = np.concatenate([[0], np.cumsum(z)]); return cs[jb] - cs[ja]
Xs = {'sign_filled>0': S_(sg * (f > 0)), 'sign_nofill': S_(sg * (f == 0)), 'sign*sqrtF': S_(sg * np.sqrt(f)),
      'sign*sqrtV': S_(sg * np.sqrt(v)), 'n_filled': S_((f > 0) * 1.0)}
for cols in [['sign*sqrtF'], ['sign_filled>0', 'sign*sqrtF'], ['sign_filled>0', 'sign_nofill', 'sign*sqrtF'], ['sign_filled>0', 'sign*sqrtF', 'sign*sqrtV']]:
    X = np.column_stack([Xs[c] for c in cols])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ b
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * res.var())
    print(regime, hs, dict(zip(cols, [f'{x:.4f}±{s:.4f}' for x, s in zip(b, se)])), 'R2 %.3f resid var/h %.4f' % (1 - res.var() / y.var(), res.var() / hs))
# residual variance vs number of filled takers
res2 = res ** 2; nt = Xs['n_filled']
df = pd.DataFrame(dict(r=res2, n=nt))
gb = df.groupby(pd.qcut(df.n, 6, duplicates='drop')).agg(r=('r', 'mean'), n=('n', 'mean'))
print(gb.round(3).to_string())
print('slope per filled taker', np.polyfit(df.n, df.r, 1).round(4))
