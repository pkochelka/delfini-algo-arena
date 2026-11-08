"""Residual reversion: E = S - (deterministic impact path incl. transient decay), sampled on a grid, shocks removed.
If xi (per-fill noise) is permanent, E is a martingale (ACF of h-increments ~ 0, VR ~ 1). Usage: resid_acf.py TAG params.json"""
import sys, json, numpy as np, pandas as pd
import prep, smooth_seg as SS
tag, pf = sys.argv[1], sys.argv[2]
P = json.load(open(prep.common.HERE / pf))
d = SS.d; r = dict(np.load(prep.common.CACHE / f'smooth_{tag}.npz'))
t = d['t']; m = r['mean']; vol = r['vol'].astype(bool); sh = r['shock'].astype(bool)
def arr(k, dflt):
    return np.where(vol, P['vol'].get(k, dflt), P['calm'].get(k, dflt)).astype(float)
shift = SS.impact(slice(0, len(t)), arr('c', 0), arr('e', 0.6), arr('pperm', 1.0), arr('tdec', 15.0))
D = np.cumsum(np.r_[0, shift[:-1]])        # deterministic path before insert i
dm = np.diff(m, prepend=m[0]); dm[sh] = 0   # remove shock jumps
import os
SH = pd.read_csv(os.environ.get('SHOCKS', prep.common.HERE / '../flow/shock_events.csv')).onset.to_numpy()
E = np.cumsum(dm) - (D - np.cumsum(np.where(sh, np.r_[0, np.diff(D)], 0)))
for h in (1, 2, 5, 10, 20, 30, 60):
    g = np.arange(0, 6000, float(h)); i = np.clip(np.searchsorted(t, g, 'right') - 1, 0, len(t) - 1)
    x = np.diff(E[i]); vv = vol[i][:-1]
    bad = np.zeros(len(x), bool)
    for a in SH: bad[max(0, int((a - h) / h)):int((a + 6) / h) + 1] = True
    ok = ~bad[:-1] & ~bad[1:]
    a1, a2 = x[:-1][ok], x[1:][ok]
    cv = ~vv[:-1][ok] & ~vv[1:][ok]
    out = [np.corrcoef(a1, a2)[0, 1], np.corrcoef(a1[cv], a2[cv])[0, 1], np.corrcoef(a1[~cv], a2[~cv])[0, 1]]
    print('h %3d  ACF1 all %.3f  calm %.3f  vol-ish %.3f   var/h calm %.4f   n %d' % (h, *out, x[:-1][ok][cv].var() / h, ok.sum()))
