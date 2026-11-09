"""Online 15-lot detector quality (j15log npz det) vs makers' episodes and the S move after detection.
uv run --with pyarrow python det15.py NPZ"""
import sys, numpy as np, pandas as pd, pathlib
R4 = pathlib.Path(__file__).resolve().parents[2]
ep = pd.read_csv(R4 / 'makers/r15_episodes.csv')
cl = pd.read_parquet(R4 / 'makers/cache/order_class.parquet', columns=['t_us', 'cls'])
sp = pd.read_parquet(R4 / 's_dynamics/s_path.parquet', columns=['t_us', 's_hat', 'vol'])
st = sp.t_us.values; sh = sp.s_hat.values
def S_at(tu): return sh[np.clip(np.searchsorted(st, tu, side='right') - 1, 0, len(sh) - 1)]
z = np.load(sys.argv[1]); d = z['det']
tu, sg, q, px, pk = d.T
a15 = set(cl.t_us.values[cl.cls.values == 2])
tp = np.array([u in a15 for u in tu])
print(f'detections {len(d)}, agent-labelled {tp.sum()} ({tp.mean():.1%}), agent orders 755; by prev_kind', np.bincount(pk.astype(int)))
print('false ones (t s, side, q):', [(round(u*1e-6,1), int(s), int(qq)) for u, s, qq in zip(tu[~tp], sg[~tp], q[~tp])][:40])
# per episode: first detection delay, count
E = ep[['t0', 't1', 'sg']].values
first = []
for a, b, g in E:
    m = (tu >= a) & (tu <= b + 1e5)
    first.append(((tu[m][0] - a) * 1e-6 if m.any() else np.nan, m.sum(), int(np.all(sg[m] == g)) if m.any() else -1))
f = np.array(first)
print('first detection delay s: median', np.nanmedian(f[:, 0]), 'max', np.nanmax(f[:, 0]), 'missed', np.isnan(f[:, 0]).sum(), 'side ok', (f[:, 2] == 1).sum())
# S move after first detection and after each detection, signed by side
for lab, sel in [('first det', None), ('every det', 'all')]:
    if sel is None:
        T0 = np.array([a + fd * 1e6 for (a, b, g), fd in zip(E, f[:, 0]) if not np.isnan(fd)]); G = np.array([g for (a, b, g), fd in zip(E, f[:, 0]) if not np.isnan(fd)])
    else:
        T0, G = tu[tp], sg[tp]
    out = [f'{h}s {np.mean((S_at(T0 + h * 1e6) - S_at(T0)) * G):+.2f}' for h in (0.5, 1, 2, 3, 5, 8, 10, 15)]
    print(lab, len(T0), ' '.join(out))
# by detection index within episode (age), S move over next 2 s / 5 s
idx = np.zeros(len(tu), int); lastt = -1e18; lasts = 0; n = 0
for i in range(len(tu)):
    if sg[i] != lasts or tu[i] - lastt > 2e6: n = 0
    n += 1; idx[i] = n; lastt = tu[i]; lasts = sg[i]
for lo, hi in [(1, 1), (2, 2), (3, 4), (5, 8), (9, 15), (16, 99)]:
    m = (idx >= lo) & (idx <= hi)
    print(f'det #{lo}-{hi}: n {m.sum():3d} S +1s {np.mean((S_at(tu[m] + 1e6) - S_at(tu[m])) * sg[m]):+.2f} +2s {np.mean((S_at(tu[m] + 2e6) - S_at(tu[m])) * sg[m]):+.2f} +5s {np.mean((S_at(tu[m] + 5e6) - S_at(tu[m])) * sg[m]):+.2f}')
# gaps between consecutive same-episode detections
g = np.diff(tu)[(np.diff(tu) < 3e6)] * 1e-6
print('gap between detections s: p50 %.2f p90 %.2f p99 %.2f max %.2f' % tuple(np.quantile(g, [.5, .9, .99, 1])))
# S move after the LAST detection of each chain (episode end proxy)
last = np.r_[np.diff(tu) > 2e6, True] & tp
print('after last det', last.sum(), ' '.join(f'{h}s {np.mean((S_at(tu[last] + h * 1e6) - S_at(tu[last])) * sg[last]):+.2f}' for h in (0.5, 1, 2, 5, 10)))
