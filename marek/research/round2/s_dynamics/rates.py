import numpy as np, pandas as pd, sys
import prep
tag = sys.argv[1] if len(sys.argv) > 1 else 'v1'
d = prep.build(); E = pd.read_csv(prep.common.CACHE / f'episodes_{tag}.csv')
t = d['t']; ioc = d['ioc']
vol = np.zeros(len(t), bool)
for a, b in zip(E.start, E.end):
    vol |= (t >= a) & (t <= b)
Tv = E.dur.sum(); Tc = 6000 - Tv
print('taker rate/s (both sides): vol %.2f calm %.2f' % ((ioc & vol).sum() / Tv, (ioc & ~vol).sum() / Tc))
print('maker rate/s: vol %.2f calm %.2f' % ((~ioc & vol).sum() / Tv, (~ioc & ~vol).sum() / Tc))
# taker buy share by regime
print('buy share vol %.3f calm %.3f' % ((d['sign'][ioc & vol] > 0).mean(), (d['sign'][ioc & ~vol] > 0).mean()))
# dispersion of taker counts per 10s in calm only
for name, msk in [('taker', ioc), ('maker', ~ioc)]:
    b = (t[msk & ~vol] // 1).astype(int)
    cnt = np.bincount(b, minlength=6000)
    # only full calm seconds
    sec = np.arange(6000)
    calm_sec = np.ones(6000, bool)
    for a, bb in zip(E.start, E.end):
        calm_sec[int(a) - 1:int(bb) + 2] = False
    c = cnt[calm_sec]
    c10 = c[:len(c) // 10 * 10].reshape(-1, 10).sum(1)
    print(name, 'calm 1s disp %.2f 10s disp %.2f mean/s %.2f' % (c.var() / c.mean(), c10.var() / c10.mean(), c.mean()))
# profile around starts and ends
for which in ['start', 'end']:
    prof_t = np.zeros(20); prof_m = np.zeros(20)
    for x in E[which]:
        for j, o in enumerate(np.arange(-20, 20, 2)):
            mm = (t >= x + o) & (t < x + o + 2)
            prof_t[j] += (mm & ioc).sum() / 2; prof_m[j] += (mm & ~ioc).sum() / 2
    print(which, 'taker/s', np.round(prof_t / len(E), 1))
    print(which, 'maker/s', np.round(prof_m / len(E), 0))
