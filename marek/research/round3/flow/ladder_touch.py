"""Ladder start vs same-side touch changes: time since last same-side touch-price change, compared with random times
(matched by regime)."""
import numpy as np, pandas as pd
from pathlib import Path
D = Path(__file__).parent; C = D / 'cache'
r = pd.read_pickle(C / 'rows.pkl'); L = pd.read_pickle(C / 'ladders.pkl')
reg = np.load(C / 'hmm_joint_2.npy')[:, 1] > 0.5
t = r.t.to_numpy()
rng = np.random.default_rng(0)
for side, col in [(0, 'bid_after'), (1, 'ask_after')]:
    px = r[col].to_numpy(); ch = np.flatnonzero(np.r_[True, np.diff(px) != 0]); tc = t[ch]
    dirn = np.r_[0, np.sign(np.diff(px[ch]))] * (1 if side == 0 else -1)  # + = touch moved toward the other side (bid up/ask down)
    l = L[L.side == side]; x = l.t0.to_numpy() - 1e-7
    rnd = np.sort(rng.uniform(0, 6000, 200000))
    for nm, sel in [('calm', ~reg[x.astype(int)]), ('vol', reg[x.astype(int)])]:
        xs = x[sel]; rs = rnd[(~reg[rnd.astype(int)]) if nm == 'calm' else reg[rnd.astype(int)]]
        k1 = np.searchsorted(tc, xs) - 1; k2 = np.searchsorted(tc, rs) - 1
        d1 = xs - tc[k1]; d2 = rs - tc[k2]
        e = np.r_[0, .002, .005, .01, .02, .05, .1, .2, .5, 1, 2, 5, 1000]
        h1 = np.histogram(d1, e)[0] / len(d1); h2 = np.histogram(d2, e)[0] / len(d2)
        print(f'side {side} {nm}: ratio ladder/random by time since last touch change', np.round(h1 / np.maximum(h2, 1e-9), 2).tolist())
        print('     last-change direction share (+ = toward other side) ladder %.3f random %.3f' % ((dirn[k1] > 0).mean(), (dirn[k2] > 0).mean()))
print('bins', [0, .002, .005, .01, .02, .05, .1, .2, .5, 1, 2, 5])
