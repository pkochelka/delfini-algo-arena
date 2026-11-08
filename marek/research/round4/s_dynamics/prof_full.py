"""Full-tape forward log-likelihood profile: vary one parameter of one regime around a params JSON.
Usage: prof_full.py params.json regime key v1 v2 ...   (regime calm|vol|both)"""
import sys, json, time
import numpy as np, pandas as pd
import smooth_seg as S
d = S.d; t = d['t']; n = len(t)
P = json.load(open(S.prep.common.HERE / sys.argv[1])); reg = sys.argv[2]; key = sys.argv[3]; vals = [float(x) for x in sys.argv[4:]]
H = S.prep.common.HERE
V = pd.read_csv(H / P['vol_segments']); hi = np.zeros(n, bool)
for a, b in zip(V.start, V.end): hi |= (t >= a) & (t < b)
SH = pd.read_csv(H / P['shock']['file']); sh = np.zeros(n, bool); wa, wb = P['shock']['win']
for a in SH.onset: sh |= (t >= a + wa) & (t < a + wb)
for v in vals:
    Q = json.loads(json.dumps(P))
    for rg in (['calm', 'vol'] if reg == 'both' else [reg]): Q[rg][key] = v
    kw = {}
    for k in set(Q['calm']) | set(Q['vol']):
        x = np.where(hi, Q['vol'].get(k, Q['calm'].get(k)), Q['calm'].get(k, Q['vol'].get(k))).astype(float)
        kw['sig2' if k == 'sig' else k] = x ** 2 if k == 'sig' else x
    kw['lamr'] = np.where(sh, Q['shock']['lamr'], kw['lamr'])
    t0 = time.time(); ll = S.loglik(slice(0, n), **kw).sum()
    print(reg, key, v, 'll %.1f' % ll, '(%.0fs)' % (time.time() - t0), flush=True)
