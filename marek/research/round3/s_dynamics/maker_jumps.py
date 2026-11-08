"""Unexplained S moves at maker steps (smoother used reset/Laplace jumps): size distribution, timing vs last taker."""
import sys, json, numpy as np
import prep, smooth_seg as SS
tag, pf = sys.argv[1], sys.argv[2]
P = json.load(open(prep.common.HERE / pf))
d = SS.d; r = dict(np.load(prep.common.CACHE / f'smooth_{tag}.npz'))
t = d['t']; m = r['mean']; vol = r['vol'].astype(bool); sh = r['shock'].astype(bool)
def arr(k, dflt): return np.where(vol, P['vol'].get(k, dflt), P['calm'].get(k, dflt)).astype(float)
shift = SS.impact(slice(0, len(t)), arr('c', 0), arr('e', 0.6), arr('pperm', 1.0), arr('tdec', 15.0))
dE = np.diff(m) - shift[:-1]
ioc = d['ioc']
ok = ~sh[:-1] & ~sh[1:] & (vol[:-1] == vol[1:]) & ~ioc[:-1]
# time since last IOC
last = np.maximum.accumulate(np.where(ioc, np.arange(len(t)), -1))
for nm, rg in (('calm', ~vol[:-1]), ('vol', vol[:-1])):
    s = np.where(ok & rg)[0]
    a = np.abs(dE[s])
    print(nm, 'maker-step |dE| quantiles 50/90/99/99.9/99.99:', np.percentile(a, [50, 90, 99, 99.9, 99.99]).round(4))
    big = s[a > 0.05]
    print('   n |dE|>0.05: %d, >0.1: %d, >0.2: %d; total dE^2 share of those >0.05: %.2f' % (len(big), (a > 0.1).sum(), (a > 0.2).sum(), (dE[big] ** 2).sum() / (dE[s] ** 2).sum()))
    ts = t[big + 1] - t[last[big]]
    print('   time since last IOC for |dE|>0.05: quantiles', np.percentile(ts, [10, 50, 90]).round(3), ' (all maker steps:', np.percentile(t[s + 1] - t[last[s]], [10, 50, 90]).round(3), ')')
    print('   share of dE^2 within 30 ms after an IOC: %.2f (time share %.2f)' % ((dE[s][t[s + 1] - t[last[s]] < 0.03] ** 2).sum() / (dE[s] ** 2).sum(), (t[s + 1] - t[last[s]] < 0.03).mean()))
    for j in big[np.argsort(-np.abs(dE[big]))][:5]:
        print('     t %.3f dE %.3f sd %.3f last ioc %.3f s ago' % (t[j], dE[j], r['sd'][j], t[j + 1] - t[last[j]]))
import pandas as pd
V = pd.read_csv(prep.common.HERE / 'vol_segments.csv'); bt = np.sort(np.r_[V.start, V.end])
SH = pd.read_csv(prep.common.HERE / '../flow/shock_events.csv').onset.to_numpy() if 'round3' in str(SS.d.get('_p', 'round3')) else None
s = np.where(ok)[0]; big = s[np.abs(dE[s]) > 0.2]
dist = np.min(np.abs(t[big][:, None] - bt[None, :]), axis=1)
print('big (>0.2) maker-step moves: n %d; within 3 s of a regime switch: %d; within 10 s: %d' % (len(big), (dist < 3).sum(), (dist < 10).sum()))
far = big[dist >= 10]
for j in far[:40]:
    print('   far: t %.3f dE %+.3f sd %.3f vol %d' % (t[j], dE[j], r['sd'][j], vol[j]))
