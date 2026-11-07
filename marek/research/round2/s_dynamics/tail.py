"""Aggressive-tail takers: what sets their price?"""
import numpy as np, pandas as pd, sys
import prep
tag = sys.argv[1] if len(sys.argv) > 1 else 'v2'
d = prep.build(); r = dict(np.load(prep.common.CACHE / f'smooth_{tag}.npz'))
t = d['t']; m = r['mean']; k = d['kind']; p = d['price']; sg = d['sign']
vol = r['vol'].astype(bool); sh = r['shock'].astype(bool)
tk = np.where((k >= 4) & (k <= 5))[0]
df = pd.DataFrame(dict(i=tk, t=t[tk], x=sg[tk] * (p[tk] - m[tk]), vol=vol[tk], sh=sh[tk], v=d['volume'][tk], f=d['filled'][tk],
                       touch=np.where(sg[tk] > 0, d['ba'][tk], d['bb'][tk]), far=np.where(sg[tk] > 0, d['bb'][tk], d['ba'][tk])))
df['p_touch'] = sg[tk] * (p[tk] - df.touch)          # ticks through the touch
df['touch_S'] = sg[tk] * (df.touch - m[tk])            # touch distance from S
df['tl'] = np.where(df.vol, df.x > 3, df.x > 1.5)
# distance to regime boundary
R = pd.read_csv(prep.common.ROOT / 'marek/research/round2/flow/regimes_envW.csv')
bnd = np.sort(np.concatenate([R.start[R.end - R.start > 3.5], R.end[R.end - R.start > 3.5]]))
j = np.searchsorted(bnd, df.t); df['dist_bnd'] = np.minimum(np.abs(df.t - bnd[np.clip(j - 1, 0, len(bnd) - 1)]), np.abs(bnd[np.clip(j, 0, len(bnd) - 1)] - df.t))
for nm, g in df[~df.sh].groupby('vol'):
    print('vol' if nm else 'calm', 'n', len(g), 'tail share %.4f' % g.tl.mean())
    print(g.groupby('tl')[['x', 'p_touch', 'touch_S', 'v', 'f', 'dist_bnd']].median().round(2).to_string())
    print('  tail share by dist to regime boundary:', g.groupby(pd.cut(g.dist_bnd, [0, 2, 5, 10, 30, 100, 2000])).tl.mean().round(3).to_dict())
    tl = g[g.tl]
    print('  tail p - touch counts:', tl.p_touch.value_counts().sort_index().head(15).to_dict())
    print('  tail x: frac part hist', np.histogram((tl.x - 0.5) % 1, bins=10)[0])
    # clustering in time: gaps between tail takers
    gp = np.diff(tl.t.to_numpy()); print('  tail gaps CV %.2f, median %.2f s' % (gp.std() / gp.mean(), np.median(gp)))
    print('  tail per 300s:', np.bincount((tl.t // 300).astype(int), minlength=20))
print('--- tail excess distributions')
c = df[~df.sh & ~df.vol & df.tl].x.to_numpy() - 1.0
v = df[~df.sh & df.vol & df.tl].x.to_numpy() - 2.5
q = [10, 25, 50, 75, 90, 95, 99]
print('calm x-1 pct', np.percentile(c, q).round(2), 'mean', c.mean().round(2))
print('vol  x-2.5 pct', np.percentile(v, q).round(2), 'mean', v.mean().round(2), ' /2.5:', (np.percentile(v, q) / 2.5).round(2))
from scipy import stats
for nm, z, lo in [('calm', c, 0.5), ('vol', v, 0.5)]:
    zz = z - lo
    print(nm, 'excess over threshold: mean %.2f sd %.2f (exp => equal)' % (zz.mean(), zz.std()), 'KS exp p=%.3f' % stats.kstest(zz, 'expon', args=(0, zz.mean())).pvalue)
