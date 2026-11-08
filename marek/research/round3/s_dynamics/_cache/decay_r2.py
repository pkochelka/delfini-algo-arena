"""Impact permanence: average sign*(S(t+tau) - S(t-)) / filled^e after filled takers, per regime, on a path TAG.
Also the same with other takers' modelled impact subtracted (cleaner)."""
import sys, numpy as np, pandas as pd
import prep
tag = sys.argv[1] if len(sys.argv) > 1 else 'np'
e = 0.6
d = prep.build(); r = dict(np.load(prep.common.CACHE / f'smooth_{tag}.npz'))
t = d['t']; m = r['mean']; vol = r['vol'].astype(bool); sh = r['shock'].astype(bool)
n = len(t)
cc = np.where(vol, 0.065, 0.0149)
imp = np.where(d['ioc'], cc * d['sign'] * d['filled'] ** e, 0.0)
C = np.cumsum(imp)  # C[j] includes taker j's impact
tk = np.where(d['ioc'] & (d['filled'] > 0))[0]
taus = [0.0, 0.05, 0.2, 0.5, 1, 2, 5, 10, 20, 60]
# exclude takers within 60 s of shock windows / regime switches
SH = pd.read_csv(prep.common.ROOT / 'marek/research/round2/flow/shock_events.csv').onset.to_numpy()
V = pd.read_csv(prep.common.HERE / 'vol_segments.csv')
bad_t = np.r_[SH, SH + 3, V.start, V.end]
def far(x, H):
    j = np.searchsorted(np.sort(bad_t), x)
    bt = np.sort(bad_t)
    lo = np.abs(x - bt[np.clip(j - 1, 0, len(bt) - 1)]); hi = np.abs(bt[np.clip(j, 0, len(bt) - 1)] - x)
    return np.minimum(lo, hi) > H
for nm, reg in (('calm', ~vol), ('vol', vol)):
    sel = tk[reg[tk] & far(t[tk], 61) & (tk > 0)]
    sel = sel[t[sel] < 5930]
    w = d['filled'][sel] ** e; sg = d['sign'][sel]
    pre = m[sel]   # S just before own impact
    out = []
    for tau in taus:
        j = np.searchsorted(t, t[sel] + tau, 'right') if tau > 0 else sel + 1
        j = np.minimum(j, n - 1)
        y = sg * (m[j] - pre)
        # subtract other takers' modelled impact between sel+1 and j-1 inclusive
        oth = C[j - 1] - C[sel]
        y2 = sg * (m[j] - pre - oth)
        out.append((tau, len(sel), (y * w).sum() / (w * w).sum(), (y2 * w).sum() / (w * w).sum(), y2.std() / np.sqrt(len(sel)) / np.sqrt((w * w).mean())))
    print('==', nm, '  tau, n, c_eff(raw), c_eff(other impact removed), se')
    for o in out: print('  %5.2f %6d %.4f %.4f %.4f' % o)
