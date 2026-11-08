"""Event-level kick residuals: per filled taker, jump J = S(next insert) - S(taker row) vs model kick; between-taker moves.
Usage: kickres.py TAG params.json"""
import sys, json, numpy as np, pandas as pd
import prep, smooth_seg as SS
tag, pf = sys.argv[1], sys.argv[2]
P = json.load(open(prep.common.HERE / pf))
d = SS.d; r = dict(np.load(prep.common.CACHE / f'smooth_{tag}.npz'))
t = d['t']; m = r['mean']; vol = r['vol'].astype(bool); sh = r['shock'].astype(bool)
n = len(t)
ti = np.flatnonzero(d['ioc'] & (d['filled'] > 0) & (np.arange(n) < n - 1))
J = m[ti + 1] - m[ti]
g = vol[ti]
c = np.where(g, P['vol']['c'], P['calm']['c']); e = np.where(g, P['vol']['e'], P['calm']['e'])
sg = d['sign'][ti]; f = d['filled'][ti]
k = sg * c * f ** e
res = J - k
# between-taker move: from after taker j to before next filled taker
B = m[np.r_[ti[1:], n - 1]] - m[ti + 1]
dtn = t[np.r_[ti[1:], n - 1]] - t[ti]
SH = pd.read_csv(prep.common.HERE / '../flow/shock_events.csv').onset.to_numpy()
bad = np.zeros(len(ti), bool)
for a in SH: bad |= (t[ti] > a - 0.5) & (t[ti] < a + 6)
for nm, gm in (('calm', ~g & ~bad), ('vol', g & ~bad)):
    x = sg[gm] * J[gm]; kk = np.abs(k[gm])
    b = np.polyfit(kk, x, 1)
    print(nm, 'n', gm.sum(), 'mean sign*J %.3f, model %.3f; slope on model kick %.3f int %.3f; sd res %.3f' % (x.mean(), kk.mean(), b[0], b[1], res[gm].std()))
    # residual vs previous residual & previous between move
    rr = res[gm]; bb = B[gm]
    print('   corr(res_j, res_j-1) %.3f  corr(res_j, B_j-1) %.3f  corr(B_j, B_j-1) %.3f  corr(B_j, res_j) %.3f' % (
        np.corrcoef(rr[1:], rr[:-1])[0, 1], np.corrcoef(rr[1:], bb[:-1])[0, 1], np.corrcoef(bb[1:], bb[:-1])[0, 1], np.corrcoef(bb, rr)[0, 1]))
    print('   var per taker: J %.3f res %.3f B %.3f ; B var / dt %.3f' % (J[gm].var(), rr.var(), bb.var(), (bb ** 2).sum() / dtn[gm].sum()))
    # residual vs sign of residual drift: does sign*res depend on sign agreement with recent move?
    mv = m[ti] - m[np.searchsorted(t, t[ti] - 2.0)]
    agree = np.sign(mv[gm]) == sg[gm]
    print('   sign*res when taker agrees with past 2s move: %.3f (n %d), disagrees %.3f (n %d)' % ((sg[gm] * rr)[agree].mean(), agree.sum(), (sg[gm] * rr)[~agree].mean(), (~agree).sum()))
    print('   res in direction of past 2s move: %.3f' % np.mean(np.sign(mv[gm]) * rr))
    qs = np.quantile(kk, [0, .2, .4, .6, .8, 1])
    for a, bq in zip(qs[:-1], qs[1:]):
        mm = (kk >= a) & (kk <= bq)
        print('   kick %.2f-%.2f: sign*J %.3f model %.3f ratio %.2f' % (a, bq, x[mm].mean(), kk[mm].mean(), x[mm].mean() / kk[mm].mean()))
