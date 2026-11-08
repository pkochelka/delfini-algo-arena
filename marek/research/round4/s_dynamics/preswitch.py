"""Kick size of fills just before / after calm->vol switches, from the no-impact-prior path (np): sign*J / f^0.62."""
import numpy as np, pandas as pd
import prep
d = prep.build(); r = dict(np.load(prep.common.CACHE / 'smooth_np.npz'))
t = d['t']; m = r['mean']; n = len(t)
ti = np.flatnonzero(d['ioc'] & (d['filled'] > 0) & (np.arange(n) < n - 2))
y = d['sign'][ti] * (m[ti + 2] - m[ti]) / d['filled'][ti] ** 0.62
V = pd.read_csv(prep.common.HERE / 'vol_segments.csv'); vol = r['vol'].astype(bool)[ti]
SH = pd.read_csv(prep.common.HERE / '../flow/shock_events.csv').onset.to_numpy()
ok = np.ones(len(ti), bool)
for a in SH: ok &= ~((t[ti] > a - 1) & (t[ti] < a + 7))
print('baseline calm %.4f (n %d)  vol %.4f (n %d)' % (y[~vol & ok].mean(), (~vol & ok).sum(), y[vol & ok].mean(), (vol & ok).sum()))
for nm, xs in (('start', V.start), ('end', V.end)):
    for a0, b0 in ((-6, -3), (-3, -1.5), (-1.5, -0.5), (-0.5, 0), (0, 0.5), (0.5, 1.5), (1.5, 4)):
        w = np.zeros(len(ti), bool)
        for x in xs: w |= (t[ti] >= x + a0) & (t[ti] < x + b0)
        w &= ok
        print('%s [%+.1f,%+.1f): n %4d  mean kick/f^.62 %.4f (se %.4f)' % (nm, a0, b0, w.sum(), y[w].mean(), y[w].std() / np.sqrt(max(w.sum(), 1))))
