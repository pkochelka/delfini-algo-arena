"""Exact change points of the per-insert vol-vs-calm llr near each switch, with a +-support interval (score within 3 nats),
and their position relative to the integer second. Usage: switch_frac.py LLRTAG"""
import sys, numpy as np, pandas as pd
import prep
d = prep.build(); t = d['t']; llr = np.load(prep.common.CACHE / f'llr_{sys.argv[1]}.npy')
V = pd.read_csv(prep.common.HERE / 'vol_segments.csv')
rows = []
for which in ('start', 'end'):
    for x0 in V[which]:
        if x0 >= 5999: continue
        i0, i1 = np.searchsorted(t, x0 - 3), np.searchsorted(t, x0 + 3)
        z = llr[i0:i1]; cs = np.r_[0, np.cumsum(z)]
        sc = (cs[-1] - cs) if which == 'start' else cs     # switch before insert j
        j = int(np.argmax(sc)); ok = np.flatnonzero(sc >= sc[j] - 3)
        tj = t[i0 + min(j, len(z) - 1)]; tlo = t[i0 + max(ok.min() - 1, 0)]; thi = t[i0 + min(ok.max(), len(z) - 1)]
        rows.append(dict(kind=which, t=tj, lo=tlo, hi=thi, frac=tj - np.round(tj), width=thi - tlo))
X = pd.DataFrame(rows)
pd.set_option('display.width', 200); pd.set_option('display.max_rows', 100)
print(X.round(3).to_string())
for k in ('start', 'end'):
    x = X[X.kind == k]
    # integer second inside [lo, hi]?
    cont = (np.floor(x.hi) >= np.ceil(x.lo))
    print(k, 'n', len(x), 'integer inside support', cont.sum(), 'mean width %.3f' % x.width.mean(), 'expected by chance %.1f' % np.minimum(x.width, 1).sum())
X.to_csv(prep.common.CACHE / 'switch_frac.csv', index=False)
