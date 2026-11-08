"""Exact regime switch times: change point of the per-insert log-lik ratio (vol vs calm observation laws given S_hat)
within +-W s of each provisional boundary. Usage: switch3.py TAG params.json init_segments.csv [W]
Writes vol_segments.csv (start,end,dur,gain_on,gain_off) and _cache/llr_TAG.npy."""
import sys, json
import numpy as np, pandas as pd
import prep
tag, pf, segf = sys.argv[1], sys.argv[2], sys.argv[3]
W = float(sys.argv[4]) if len(sys.argv) > 4 else 8.0
P = json.load(open(prep.common.HERE / pf))
d = prep.build(); r = dict(np.load(prep.common.CACHE / f'smooth_{tag}.npz'))
t = d['t']; k = d['kind']; p = d['price']; sg = d['sign']


def llr_arr(S):
    def Fd(z, d0, th): return np.where(z > d0, 1 - np.exp(-(z - d0) / th), 0.0)
    z = np.where(sg > 0, S - p, p - S)
    x = sg * (p - S)
    L = {}
    for nm in ('calm', 'vol'):
        q = P[nm]
        mk = np.where(k <= 1, Fd(z, q['d0'], q['theta']) - Fd(z - 1, q['d0'], q['theta']), Fd(z, q['d0'], q['theta']))
        off, eps, tau = q['off'], q['eps'], q['tau']
        tl = np.where(x > off + 0.5, eps / tau * np.exp(-(x - off - 0.5) / tau), 0)
        tkl = np.where(np.abs(x - off) <= 0.5, 1 - eps, 0) + tl
        L[nm] = np.log(np.where(k <= 3, mk, tkl) + 1e-4)
    return L['vol'] - L['calm']


sh = r['shock'].astype(bool)
llr = llr_arr(r['mean']); llr[sh] = 0.0
for a in pd.read_csv(prep.common.HERE / '../flow/shock_events.csv').onset:   # whole drought + jump: few, atypical makers
    llr[(t >= a - 0.1) & (t < a + 5.0)] = 0.0
llr[np.load(prep.common.HERE / '../flow/cache/ladder_member.npy')[:, 0] >= 0] = 0.0
np.save(prep.common.CACHE / f'llr_{tag}.npy', llr)
R = pd.read_csv(prep.common.HERE / segf)
rows = []
for a, b in zip(R.start, R.end):
    res = []
    for which, x0 in (('on', a), ('off', b)):
        i0, i1 = np.searchsorted(t, x0 - W), np.searchsorted(t, x0 + W)
        z = llr[i0:i1]; cs = np.concatenate([[0], np.cumsum(z)])
        score = (cs[-1] - cs) if which == 'on' else cs
        j = int(np.argmax(score)); jj = i0 + min(j, len(z) - 1)
        res.append((t[jj], score[j] - max(score[0], score[-1])))
    rows.append(dict(start=res[0][0], end=res[1][0], gain_on=res[0][1], gain_off=res[1][1]))
X = pd.DataFrame(rows); X['dur'] = X.end - X.start
# whole-segment evidence
X['seg_llr'] = [llr[np.searchsorted(t, a):np.searchsorted(t, b)].sum() for a, b in zip(X.start, X.end)]
pd.set_option('display.width', 200)
print(X.round(3).to_string())
X.to_csv(prep.common.HERE / 'vol_segments.csv', index=False)
