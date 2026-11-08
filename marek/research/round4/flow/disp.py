"""Taker count dispersion within regime interiors, excluding shock windows (-3..+25 s)."""
import numpy as np, pandas as pd
from pathlib import Path
D = Path(__file__).parent; C = D / 'cache'
z = np.load(C / 'bins.npz'); vol = np.load(C / 'vol1s.npy'); ev = pd.read_csv(D / 'shock_events.csv')
ok = np.ones(6000, bool)
for o in ev.onset: ok[max(0, int(o) - 3):int(o) + 25] = False
for reg in [0, 1]:
    good = ok & (vol == reg)
    for W in [1, 5, 10, 30]:
        tb = np.convolve(z['tb1'], np.ones(W), 'valid'); ts = np.convolve(z['ts1'], np.ones(W), 'valid')
        mm = np.convolve(good, np.ones(W), 'valid') == W
        tot = (tb + ts)[mm]; im = (tb - ts)[mm]
        print(f'reg {reg} W {W:3d} n {mm.sum():5d} total mean {tot.mean():7.2f} DI {tot.var() / tot.mean():.3f}  imb var/total {im.var() / tot.mean():.3f}  buy DI {tb[mm].var() / tb[mm].mean():.3f}')
