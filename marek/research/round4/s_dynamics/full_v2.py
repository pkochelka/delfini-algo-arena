"""Full-tape smoother with regime-dependent parameters.
Usage: full_v2.py TAG params.json
params.json: {"calm": {...}, "vol": {...}, "shock": {"lamr": ..}}  keys as in smooth_seg.DEFAULT (sig, not sig2)
Regimes: P["vol_segments"] csv (start,end) if given else all calm; shock windows P["shock"]["win"] after onsets in P["shock"]["file"].
"""
import sys, json, time
import numpy as np, pandas as pd
import smooth_seg as S

d = S.d; t = d['t']; n = len(t)
tag = sys.argv[1]; P = json.load(open(sys.argv[2]))


def labels():
    H = S.prep.common.HERE
    SH = pd.read_csv(H / P['shock'].get('file', '_cache/droughts.csv'))
    hi = np.zeros(n, bool)
    vf = P.get('vol_segments')
    if vf:
        V = pd.read_csv(H / vf)  # columns start,end
        for a, b in zip(V.start, V.end):
            hi |= (t >= a) & (t < b)
    sh = np.zeros(n, bool)
    wa, wb = P['shock'].get('win', [2.7, 3.6])
    for a in SH[P['shock'].get('col', 'onset')]:
        sh |= (t >= a + wa) & (t < a + wb)
    return hi, sh, SH


hi, sh, SH = labels()
kw = {}
keys = set(P['calm']) | set(P['vol'])
for k in keys:
    v = np.where(hi, P['vol'].get(k, P['calm'].get(k)), P['calm'].get(k, P['vol'].get(k))).astype(float)
    if k == 'sig':
        kw['sig2'] = v ** 2
    else:
        kw[k] = v
kw['lamr'] = np.where(sh, P['shock']['lamr'], kw.get('lamr', np.full(n, 0.003)))
CH = 60000; OV = 3000
keys_o = ['mean', 'sd', 'lo', 'med', 'hi', 'fmean', 'fsd', 'flo', 'fhi', 'll', 'imp']
out = {k: np.zeros(n) for k in keys_o}
t0 = time.time()
for s in range(0, n, CH):
    a, b = max(0, s - OV), min(n, s + CH + OV)
    r = S.smooth(slice(a, b), **kw)
    e = min(n, s + CH)
    for k in keys_o:
        out[k][s:e] = r[k][s - a:e - a]
print('time %.0f total ll %.1f' % (time.time() - t0, out['ll'].sum()))
out['vol'] = hi; out['shock'] = sh
np.savez(S.prep.common.CACHE / f'smooth_{tag}.npz', **out)
