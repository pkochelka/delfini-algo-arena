"""Why are volatile maker counts overdispersed? Dispersion by size band, by window; excitation after takers."""
import numpy as np, pandas as pd
from pathlib import Path
D = Path(__file__).parent; C = D / 'cache'
r = pd.read_pickle(C / 'rows.pkl'); ins = r[r.op == 0]
mk = ins[~ins.ioc]; tk = ins[ins.ioc]
reg = (np.load(C / 'hmm_joint_2.npy')[:, 1] > .5)
mask = np.load(C / 'bins1s.npz')['mask']
sec = mk.t.to_numpy().astype(int); v = mk.vol.to_numpy()
for lo, hi in [(1, 10), (11, 19), (20, 70), (71, 89), (90, 320), (321, 10**9), (1, 10**9)]:
    s = (v >= lo) & (v <= hi)
    for w in [0.1, 1, 10]:
        nb = int(6000 / w); c = np.bincount((mk.t.to_numpy()[s] / w).astype(int), minlength=nb)[:nb]
        rr = np.repeat(reg, int(round(1 / w))) if w < 1 else reg.reshape(-1, int(w)).mean(1) > .99 if w > 1 else reg
        mm = np.repeat(mask, int(round(1 / w))) if w < 1 else mask.reshape(-1, int(w)).all(1) if w > 1 else mask
        rc = rr if w <= 1 else rr
        cc = (reg.reshape(-1, int(w)).mean(1) < .01) if w > 1 else ~rr
        out = []
        for nm, sel in [('calm', cc & mm), ('vol', rr & mm)]:
            out.append(f'{nm} rate {c[sel].mean() / w:7.2f} DI {c[sel].var() / max(c[sel].mean(), 1e-9):5.2f}')
        print(f'size {lo}-{hi} w={w}: ' + ' | '.join(out))
