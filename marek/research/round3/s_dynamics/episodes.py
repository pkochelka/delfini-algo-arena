"""Per-episode homogeneity: for each vol episode and the calm gap after it: non-ladder maker gap quantiles, taker offset
quantiles, taker rate, impact proxy. Path TAG."""
import sys, numpy as np, pandas as pd
import prep
tag = sys.argv[1]
d = prep.build(); r = dict(np.load(prep.common.CACHE / f'smooth_{tag}.npz'))
t = d['t']; k = d['kind']; p = d['price']; sg = d['sign']; m = r['mean']
lad = np.load(prep.common.HERE / '../flow/cache/ladder_member.npy')[:, 0] >= 0
V = pd.read_csv(prep.common.HERE / 'vol_segments.csv')
SH = pd.read_csv(prep.common.HERE / '../flow/shock_events.csv').onset.to_numpy()
bad = np.zeros(len(t), bool)
for a in SH: bad |= (t >= a - 0.1) & (t < a + 5)
bounds = [(0.0, V.start[0], 'calm')]
for j in range(len(V)):
    bounds.append((V.start[j], V.end[j], 'vol'))
    bounds.append((V.end[j], V.start[j + 1] if j + 1 < len(V) else 6000.0, 'calm'))
rows = []
for a, b, nm in bounds:
    s = (t >= a + 0.5) & (t < b - 0.5) & ~bad
    mk = s & (k <= 1) & ~lad
    tkm = s & (k >= 4)
    g = sg[mk] * (m[mk] - p[mk]); x = sg[tkm] * (p[tkm] - m[tkm])
    if mk.sum() < 50 or tkm.sum() < 10: continue
    rows.append(dict(a=a, b=b, reg=nm, dur=b - a, gap_min=g.min(), gap_p01=np.percentile(g, 1), gap_med=np.median(g),
                     x_p05=np.percentile(x, 5), x_med=np.median(x), x_p90=np.percentile(x, 90), tk_rate=tkm.sum() / (b - a),
                     mk_rate=(s & (k <= 3)).sum() / (b - a)))
X = pd.DataFrame(rows)
pd.set_option('display.width', 220); pd.set_option('display.max_rows', 200)
print(X.round(2).to_string())
print(X.groupby('reg')[['gap_min', 'gap_p01', 'gap_med', 'x_p05', 'x_med', 'x_p90', 'tk_rate', 'mk_rate']].describe().T.round(2).to_string())
