"""How predictable is S? Regress future S change (smoothed path) over h on the causal Kalman state (filtered velocity D and
transient Y) at taker times. Usage: predict.py TAG KFSTATE.npz"""
import sys, numpy as np, pandas as pd
import prep
tag, kfs = sys.argv[1], sys.argv[2]
d = prep.build(); r = dict(np.load(prep.common.CACHE / f'smooth_{tag}.npz'))
t = d['t']; m = r['mean']; vol = r['vol'].astype(bool)
z = np.load(kfs); tt = z['t']; io = z['io']; vf = z['filt'][:, 2]; yf = z['filt'][:, 1]
SH = pd.read_csv(prep.common.HERE / '../flow/shock_events.csv').onset.to_numpy()
def at(x): return m[np.clip(np.searchsorted(t, x, 'right') - 1, 0, len(t) - 1)]
for h in (0.5, 1, 2, 5, 10, 30):
    ok = tt < 6000 - h
    for a in SH: ok &= ~((tt > a - h - 0.5) & (tt < a + 6))
    y = at(tt + h) - m[io]
    for nm, w in (('calm', ok & ~vol[io]), ('vol', ok & vol[io])):
        X = np.c_[vf[w], yf[w], np.ones(w.sum())]
        b, *_ = np.linalg.lstsq(X, y[w], rcond=None); res = y[w] - X @ b
        print('h %4.1f %-4s: dS sd %6.2f  R2 %.3f  beta_v %.3f (OU tau-implied %.2f)  beta_Y %+.3f' % (
            h, nm, y[w].std(), 1 - res.var() / y[w].var(), b[0], 0, b[1]))
