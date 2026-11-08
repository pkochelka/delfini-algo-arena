"""Exact regime switch times: per-insert log-likelihood ratio vol vs calm (maker D law + taker offset law) given S_hat,
change point within +-8 s of each envW boundary. Also a causal detector and its latency."""
import numpy as np, pandas as pd, sys
import prep
tag = sys.argv[1] if len(sys.argv) > 1 else 'v2'
d = prep.build(); r = dict(np.load(prep.common.CACHE / f'smooth_{tag}.npz'))
t = d['t']; k = d['kind']; p = d['price']; sg = d['sign']
def llr_arr(S):
    """log L_vol - log L_calm per insert given S."""
    def Fd(z, d0, th): return np.where(z > d0, 1 - np.exp(-(z - d0) / th), 0.0)
    out = np.zeros(len(S))
    z = np.where(sg > 0, S - p, p - S)
    L = {}
    for nm, d0, th, off, eps in [('c', 0.05, 0.45, 1.0, 0.05), ('v', 0.70, 0.89, 2.5, 0.10)]:
        mk = np.where(k <= 1, Fd(z, d0, th) - Fd(z - 1, d0, th), Fd(z, d0, th))
        x = sg * (p - S)  # taker: x in [off-.5, off+.5]
        tl = np.where(x > off + 0.5, eps / 3.0 * np.exp(-(x - off - 0.5) / 3.0), 0)
        tkl = np.where(np.abs(x - off) <= 0.5, 1 - eps, 0) + tl
        L[nm] = np.log(np.where(k <= 3, mk, tkl) + 1e-4)
    return L['v'] - L['c']
llr = llr_arr(r['mean'])
R = pd.read_csv(prep.common.ROOT / 'marek/research/round2/flow/regimes_envW.csv')
R = R[(R.end - R.start) > 3.5].reset_index(drop=True)
rows = []
for a, b, hh in zip(R.start, R.end, R.high):
    if not hh: continue
    for which, x0 in [('on', a), ('off', b)]:
        i0, i1 = np.searchsorted(t, x0 - 8), np.searchsorted(t, x0 + 8)
        z = llr[i0:i1]; cs = np.concatenate([[0], np.cumsum(z)])
        # on: calm before, vol after -> maximise sum after
        score = (cs[-1] - cs) if which == 'on' else cs
        j = int(np.argmax(score))
        jj = i0 + min(j, len(z) - 1)
        rows.append(dict(kind=which, env=x0, t_switch=t[jj], gain=score[j] - max(score[0], score[-1])))
X = pd.DataFrame(rows)
print(X.round(3).to_string())
on = X[X.kind == 'on'].t_switch.to_numpy(); off = X[X.kind == 'off'].t_switch.to_numpy()
dur = off - on; gaps = on[1:] - off[:-1]
print('vol durations', np.round(np.sort(dur), 1), 'mean %.1f' % dur.mean())
print('calm durations', np.round(np.sort(gaps), 1), 'mean %.1f (plus first %.1f, last %.1f)' % (gaps.mean(), on[0], 6000 - off[-1]))
X.to_csv(prep.common.CACHE / 'switch_times.csv', index=False)
np.save(prep.common.CACHE / 'llr.npy', llr)
# causal detector using the *filtered* S: CUSUM on llr
llr_f = llr_arr(r['fmean'])
lat = []
for a, b in zip(on, off):
    i0 = np.searchsorted(t, a - 0.5)
    s = 0; hit = None
    for i in range(i0, i0 + 4000):
        s = max(0, s + llr_f[i])
        if s > 10: hit = t[i]; break
    lat.append(None if hit is None else hit - a)
print('causal CUSUM(h=10) detection latency after onset (s):', np.round([x for x in lat if x is not None], 3))
# false alarm rate in calm
s = 0; fa = 0; calm_time = 0
vol_mask = np.zeros(len(t), bool)
for a, b in zip(on, off): vol_mask |= (t >= a - 1) & (t < b + 3)
for i in np.where(~vol_mask)[0][::1]:
    s = max(0, s + llr_f[i])
    if s > 10: fa += 1; s = 0
print('false alarms in calm:', fa)
