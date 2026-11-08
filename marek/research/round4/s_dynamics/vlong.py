"""Long-lag structure of residual increments dE (1 s grid) inside volatile episodes; per-episode drift."""
import sys, json, numpy as np, pandas as pd
import prep, smooth_seg as SS
tag, pf = sys.argv[1], sys.argv[2]
P = json.load(open(prep.common.HERE / pf))
d = SS.d; r = dict(np.load(prep.common.CACHE / f'smooth_{tag}.npz'))
t = d['t']; m = r['mean']; vol = r['vol'].astype(bool)
def arr(k, dflt): return np.where(vol, P['vol'].get(k, dflt), P['calm'].get(k, dflt)).astype(float)
d['vshift'][:] = 0.0
shift = SS.impact(slice(0, len(t)), arr('c', 0), arr('e', 0.6), arr('pperm', 1.0), arr('tdec', 15.0))
I = np.cumsum(np.r_[0, shift[:-1]]); E = m - I
SH = pd.read_csv(prep.common.HERE / '../flow/shock_events.csv').onset.to_numpy()
V = pd.read_csv(prep.common.HERE / 'vol_segments.csv')
B = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
g = np.arange(0, 6000, B); i = np.clip(np.searchsorted(t, g, 'right') - 1, 0, len(t) - 1)
y = np.diff(E[i]); yi = np.diff(I[i]); ys = np.diff(m[i])
bad = np.zeros(len(y), bool)
for a in SH: bad[int((a - 0.5) / B):int((a + 6) / B) + 1] = True
seg = np.full(len(y), -1)
for k, (a, b) in enumerate(zip(V.start, V.end)): seg[int(np.ceil(a / B)) + 1:int(b / B) - 1] = k
cseg = np.full(len(y), -1); edges = np.r_[0, V.end, 6000]; starts = np.r_[0, V.end]; ends = np.r_[V.start, 6000]
for k, (a, b) in enumerate(zip(starts, ends)): cseg[int(np.ceil(a / B)) + 1:int(b / B) - 1] = k
for nm, sg in (('vol', seg), ('calm', cseg)):
    ok = (sg >= 0) & ~bad
    x = np.where(ok, y - y[ok].mean(), 0); v0 = (x[ok] ** 2).mean()
    L = [1, 2, 3, 5, 7, 10, 15, 20, 30, 45, 60]
    ac = []
    for l in L:
        l = int(l / B); k = np.arange(len(y) - l); w = ok[k] & ok[k + l] & (sg[k] == sg[k + l])
        ac.append((x[k] * x[k + l])[w].mean() / v0)
    print(nm, 'ACF dE (B %.1f):' % B, ' '.join('%g:%.3f' % (l, a) for l, a in zip(L, ac)))
# per vol episode: net E, net impact, duration
rows = []
for k, (a, b) in enumerate(zip(V.start, V.end)):
    w = (seg == k) & ~bad
    if w.sum() < 5: continue
    rows.append(dict(start=a, dur=w.sum() * B, netE=y[w].sum(), netI=yi[w].sum(), netS=ys[w].sum(),
                     half1=y[w][:w.sum() // 2].sum(), half2=y[w][w.sum() // 2:].sum()))
X = pd.DataFrame(rows)
X['vE'] = X.netE / X.dur
print(X.round(1).to_string())
print('corr(half1, half2) of E drift within vol episodes: %.3f' % np.corrcoef(X.half1 / X.dur, X.half2 / X.dur)[0, 1])
print('per-episode mean velocity sd %.2f; OU prediction sd(netE/dur) for sd s, tau: sqrt(2 s^2 tau / dur)' % X.vE.std())
