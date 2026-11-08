"""Ladder start intensity conditional on book state (spread, touch qty) and on recent ladders (self-excitation)."""
import numpy as np, pandas as pd
from pathlib import Path
D = Path(__file__).parent; C = D / 'cache'
r = pd.read_pickle(C / 'rows.pkl'); L = pd.read_pickle(C / 'ladders.pkl')
reg = (np.load(C / 'hmm_joint_2.npy')[:, 1] > .5)
L = L[L.steps.apply(lambda x: all(s == 1 for s in x))]
t = r.t.to_numpy(); dur = np.diff(t, append=6000.0)
spr = (r.ask_after - r.bid_after).to_numpy()
R = reg[np.minimum(t.astype(int), 5999)]
li = np.searchsorted(t, L.t0.to_numpy()) - 1  # row before ladder head
for nm, sel in [('calm', ~R), ('vol', R)]:
    print('==', nm)
    for s_ in range(1, 7):
        m = sel & (spr == s_); n = (sel[li] & (spr[li] == s_)).sum()
        if dur[m].sum() > 5: print(f'  spread {s_}: time {dur[m].sum():7.1f}s ladders {n:6d} rate {n / dur[m].sum():6.2f}/s')
# self-excitation of ladder starts
x = np.sort(L.t0.to_numpy())
for nm, sel in [('calm', ~reg[x.astype(int)]), ('vol', reg[x.astype(int)])]:
    s = x[sel & (x > 10) & (x < 5990)]
    base = (np.searchsorted(x, s - 3) - np.searchsorted(x, s - 6) + np.searchsorted(x, s + 6) - np.searchsorted(x, s + 3)) / 6
    out = []
    for a, b in [(0.0025, 0.01), (0.01, 0.05), (0.05, 0.2), (0.2, 0.5), (0.5, 1), (1, 2)]:
        n = np.searchsorted(x, s + b) - np.searchsorted(x, s + a, 'right'); out.append(round(n.sum() / (base.sum() * (b - a)), 2))
    print(nm, 'ladder->ladder intensity ratio', out)
