"""Excitation tests per regime: intensity of dst stream in windows after src events vs local baseline (±3–6 s).
Maker streams split by size band (S=1-19, M=20-70, L=90-320, X other)."""
import numpy as np, pandas as pd
from pathlib import Path
D = Path(__file__).parent; C = D / 'cache'
r = pd.read_pickle(C / 'rows.pkl'); ev = pd.read_csv(D / 'shock_events.csv')
reg = (np.load(C / 'hmm_joint_2.npy')[:, 1] > .5)
ins = r[r.op == 0]
t_all = ins.t.to_numpy(); ok = np.ones(len(ins), bool)
for o in ev.onset: ok &= ~((t_all > o - 2) & (t_all < o + 20))
I = ins[ok]; v = I.vol.to_numpy(); ioc = I.ioc.to_numpy(); t = I.t.to_numpy()
band = np.select([ioc, v <= 19, v <= 70, (v >= 90) & (v <= 320)], ['T', 'S', 'M', 'L'], 'X')
win = [(0, 0.005), (0.005, 0.02), (0.02, 0.05), (0.05, 0.2), (0.2, 1)]
def ratio(s, dst):
    s = s[(s > 10) & (s < 5990)]; out = []
    base = (np.searchsorted(dst, s - 3) - np.searchsorted(dst, s - 6) + np.searchsorted(dst, s + 6) - np.searchsorted(dst, s + 3)) / 6
    for a, b in win:
        n = np.searchsorted(dst, s + b) - np.searchsorted(dst, s + a, 'right')
        out.append(n.sum() / (base.sum() * (b - a)))
    return out
print('windows', win)
for nm, R in [('calm', False), ('vol', True)]:
    inR = reg[t.astype(int)] == R
    # require whole ±6 s in regime
    sec = t.astype(int); good = np.array([reg[max(0, s - 7):s + 7].all() if R else (~reg[max(0, s - 7):s + 7]).all() for s in range(6000)])
    print('=====', nm)
    for a in 'TSML':
        src = t[(band == a) & good[sec]]
        for b in 'TSMLX':
            dst = t[band == b]
            print(f'  {a}->{b}', np.round(ratio(src, dst), 2))
