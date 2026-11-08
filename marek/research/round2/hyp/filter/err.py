import sys, json, numpy as np
L = np.load(sys.argv[1]); seed = int(sys.argv[2])
tr = json.load(open(f'data/synth/r2_v2_seed{seed}.truth.json'))
t, F, sd, pv, exc, S, dr, hot, pos = L.T
e = F - S
vol = np.zeros(len(t), bool)
for a, b, v in tr['segs']:
    if v: vol |= (t >= a) & (t < b)
since = np.full(len(t), 1e9)
for o in tr['onsets']:
    j = o + 3.0
    m = (t >= j); since[m] = np.minimum(since[m], t[m] - j)
pre = np.zeros(len(t), bool)
for o in tr['onsets']: pre |= (t >= o) & (t < o + 3.0)
def rep(name, m):
    if m.sum() == 0: return
    x = e[m]
    print(f'{name:28s} n={m.sum():7d} rmse={np.sqrt((x**2).mean()):.3f} mae={np.abs(x).mean():.3f} bias={x.mean():+.3f} '
          f'p95={np.quantile(np.abs(x),.95):.2f} sd_med={np.median(sd[m]):.3f} z_rms={np.sqrt(((x/np.maximum(sd[m],1e-3))**2).mean()):.2f} pv={pv[m].mean():.2f}')
calm = ~vol & ~pre & (since > 5)
rep('all', np.ones(len(t), bool))
rep('calm (no shock +-)', calm)
rep('volatile', vol & ~pre & (since > 5))
rep('drought (pre-jump)', pre)
for a, b in [(0, .1), (.1, .3), (.3, 1), (1, 2), (2, 5), (5, 15)]:
    rep(f'after shock {a}-{b}s', (since >= a) & (since < b))
# vol entry: first seconds after calm->vol
ent = np.full(len(t), 1e9)
for a, b, v in tr['segs']:
    if v and a > 0:
        m = (t >= a) & (t < b); ent[m] = t[m] - a
for a, b in [(0, 2), (2, 5), (5, 20), (20, 1e8)]:
    rep(f'vol since entry {a}-{b}s', (ent >= a) & (ent < b) & ~pre)
ex = np.full(len(t), 1e9)
for a, b, v in tr['segs']:
    if not v and a > 0:
        m = (t >= a) & (t < b); ex[m] = t[m] - a
for a, b in [(0, 5), (5, 20)]:
    rep(f'calm since exit {a}-{b}s', (ex >= a) & (ex < b) & ~pre)
# regime classification
print('pv>0.5 when vol', (pv[vol] > .5).mean(), 'pv>0.5 when calm', (pv[~vol] > .5).mean())
