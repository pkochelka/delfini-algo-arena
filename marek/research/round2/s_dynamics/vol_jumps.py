"""Residual (non-impact) S moves in v3: per regime, distribution of X steps across takers vs between takers."""
import numpy as np, pandas as pd, sys, json
import prep
tag = sys.argv[1] if len(sys.argv) > 1 else 'v3'
d = prep.build(); r = dict(np.load(prep.common.CACHE / f'smooth_{tag}.npz'))
t = d['t']; m = r['mean']; vol = r['vol'].astype(bool); sh = r['shock'].astype(bool)
dX = np.diff(m) - r['imp'][:-1]
ioc = d['ioc'][:-1] & (d['filled'][:-1] > 0)
for nm, reg in [('calm', ~vol & ~sh), ('vol', vol & ~sh)]:
    rg = reg[:-1] & reg[1:]
    a = dX[rg & ioc]; b = dX[rg & ~ioc]
    T = (t[1:] - t[:-1])[rg].sum()
    print(nm, 'across filled takers: n %d sd %.3f kurt %.1f |>1| %d ; between: sum var/time %.4f /s, |>1| %d, per s %.3f' % (
        len(a), a.std(), pd.Series(a).kurt(), (np.abs(a) > 1).sum(), (b ** 2).sum() / T, (np.abs(b) > 1).sum(), (np.abs(b) > 1).sum() / T))
