"""Where does residual variance enter? Per-insert residual increment dE_i = E[i+1]-E[i] (E = S_hat - deterministic path),
mean dE^2 by event type and regime, plus sum over IOC-to-IOC intervals. Usage: resid_event.py TAG params.json"""
import sys, json, numpy as np
import prep, smooth_seg as SS
tag, pf = sys.argv[1], sys.argv[2]
P = json.load(open(prep.common.HERE / pf))
d = SS.d; r = dict(np.load(prep.common.CACHE / f'smooth_{tag}.npz'))
t = d['t']; m = r['mean']; vol = r['vol'].astype(bool); sh = r['shock'].astype(bool)
def arr(k, dflt): return np.where(vol, P['vol'].get(k, dflt), P['calm'].get(k, dflt)).astype(float)
shift = SS.impact(slice(0, len(t)), arr('c', 0), arr('e', 0.6), arr('pperm', 1.0), arr('tdec', 15.0))
dE = np.diff(m) - shift[:-1]
ioc = d['ioc'][:-1]; f = d['filled'][:-1]
ok = ~sh[:-1] & ~sh[1:] & (vol[:-1] == vol[1:])
for nm, rg in (('calm', ~vol[:-1]), ('vol', vol[:-1])):
    s = ok & rg
    a = s & ioc & (f > 0); b = s & ioc & (f == 0); c = s & ~ioc
    print(nm, 'filled taker: n %d mean dE %.4f mean dE^2 %.5f | unfilled: n %d mean dE^2 %.5f | maker: n %d mean dE^2 %.6f sum share %.2f' % (
        a.sum(), (dE[a] * d['sign'][:-1][a]).mean(), (dE[a] ** 2).mean(), b.sum(), (dE[b] ** 2).mean(), c.sum(), (dE[c] ** 2).mean(),
        (dE[c] ** 2).sum() / (dE[s] ** 2).sum()))
    # per time: total residual var rate
    T = (np.diff(t)[s]).sum()
    print('    rate of sum dE^2: %.4f /s; filled takers %.4f /s; makers %.4f /s' % ((dE[s] ** 2).sum() / T, (dE[a] ** 2).sum() / T, (dE[c] ** 2).sum() / T))
