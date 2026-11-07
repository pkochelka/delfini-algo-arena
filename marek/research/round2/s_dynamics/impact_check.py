"""Across-taker S change vs sign*sqrt(filled), by regime (flow agent's envW regimes), from a smoothed S path."""
import numpy as np, pandas as pd, sys
import prep
tag = sys.argv[1] if len(sys.argv) > 1 else 'v1'
d = prep.build(); r = dict(np.load(prep.common.CACHE / f'smooth_{tag}.npz'))
t = d['t']; m = r['mean']
R = pd.read_csv(prep.common.ROOT / 'marek/research/round2/flow/regimes_envW.csv')
SH = pd.read_csv(prep.common.ROOT / 'marek/research/round2/flow/shock_events.csv')
hi = np.zeros(len(t), bool)
for a, b, hh in zip(R.start, R.end, R.high):
    if hh: hi |= (t >= a) & (t < b)
shock = np.zeros(len(t), bool)
for a in SH.onset: shock |= (t >= a - 0.5) & (t < a + 6)
tk = np.where(d['ioc'])[0]
tk = tk[(tk > 0) & (tk < len(t) - 1)]
# S just before = mean at event (pre-impact), S after = mean at the next event (post-impact)
pre = m[tk]; post = m[tk + 1]
x = d['sign'][tk] * np.sqrt(d['filled'][tk]); xo = d['sign'][tk] * np.sqrt(d['volume'][tk])
ok = (r['sd'][tk] < 0.3) & (r['sd'][tk + 1] < 0.3)
for name, msk in [('calm', ~hi[tk] & ~shock[tk]), ('vol', hi[tk] & ~shock[tk])]:
    mm = msk & ok
    y = post[mm] - pre[mm]
    for nm, z in [('filled', x[mm]), ('ordered', xo[mm])]:
        c = (z * y).sum() / (z * z).sum()
        print(name, nm, 'n', mm.sum(), 'c %.4f' % c, 'R2 %.3f' % (1 - ((y - c * z) ** 2).sum() / (y ** 2).sum()))
    # exponent scan
    f = d['filled'][tk][mm]; sg = d['sign'][tk][mm]
    for e in [0.3, 0.4, 0.5, 0.6, 0.7, 1.0]:
        z = sg * f ** e; c = (z * y).sum() / (z * z).sum()
        print('   e', e, 'c %.4f R2 %.3f' % (c, 1 - ((y - c * z) ** 2).sum() / (y ** 2).sum()))
