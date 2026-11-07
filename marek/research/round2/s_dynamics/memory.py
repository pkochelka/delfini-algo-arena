"""Momentum / mean reversion / drift checks on S (shock jumps removed) and on X = S - impact."""
import numpy as np, pandas as pd, sys
import prep
tag = sys.argv[1] if len(sys.argv) > 1 else 'v2'
c_c, c_v, e = [float(x) for x in sys.argv[2:5]] if len(sys.argv) > 4 else (0.0143, 0.0715, 0.625)
d = prep.build(); r = dict(np.load(prep.common.CACHE / f'smooth_{tag}.npz'))
t = d['t']; m = r['mean']; vol = r['vol'].astype(bool); sh = r['shock'].astype(bool)
# remove shock jumps: S change across shock windows
dm = np.diff(m, prepend=m[0]); dm[sh] = 0
Sx = np.cumsum(dm) + m[0]
cc = np.where(vol, c_v, c_c)
imp = np.where(d['ioc'], cc * d['sign'] * d['filled'] ** e, 0)
X = Sx - (np.cumsum(imp) - imp)
g = np.arange(0, 6000, 1.0); i = np.clip(np.searchsorted(t, g, 'right') - 1, 0, len(t) - 1)
volg = vol[i]
for nm, A in [('S (no shocks)', Sx[i]), ('X = S - impact', X[i])]:
    print('==', nm)
    for h in [1, 10, 60, 300]:
        dd = A[h::h] - A[:-h:h]
        vr = dd.var() / h / ((A[1:] - A[:-1]).var())
        ac = np.corrcoef(dd[:-1], dd[1:])[0, 1]
        print('  h %4d var/h %.3f  VR(h vs 1s) %.2f  ACF1 of h-increments %.3f  mean/h %.4f' % (h, dd.var() / h, vr, ac, dd.mean() / h))
# calm-only and vol-only increment means and ACF at 1 s / 10 s
d1 = np.diff(X[i]); v1 = volg[:-1] & volg[1:]; c1 = ~volg[:-1] & ~volg[1:]
for nm, mm in [('calm', c1), ('vol', v1)]:
    z = d1[mm]
    print(nm, '1s X incr: mean %.4f sd %.3f, t-stat drift %.2f' % (z.mean(), z.std(), z.mean() / z.std() * np.sqrt(len(z))))
# impact-free predictability: does past 10s S move predict next 10s X move?
h = 10; A = Sx[i]; B = X[i]
past = A[h:-h] - A[:-2 * h]; fut = B[2 * h:] - B[h:-h]
print('corr(past 10s S move, next 10s X move) %.3f  (n=%d overlapping)' % (np.corrcoef(past, fut)[0, 1], len(past)))
past = A[60:-60] - A[:-120]; fut = B[120:] - B[60:-60]
print('corr(past 60s S move, next 60s X move) %.3f' % np.corrcoef(past, fut)[0, 1])
# mean reversion to a level: regress 60s X increment on (S - S 300s-MA)
ma = pd.Series(A).rolling(300).mean().to_numpy()
dev = (A - ma)[300:-60]; fut = (B[360:] - B[300:-60])
print('corr(S - MA300, next 60s X) %.3f' % np.corrcoef(dev, fut)[0, 1])
