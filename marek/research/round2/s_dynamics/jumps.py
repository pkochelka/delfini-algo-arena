import numpy as np, pandas as pd, sys
import prep
d = prep.build(); tag = sys.argv[1] if len(sys.argv) > 1 else 'v1'
thr = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
r = dict(np.load(prep.common.CACHE / f'smooth_{tag}.npz'))
t = d['t']; m = r['mean']
imp = np.where(d['ioc'], 0.026 * d['sign'] * np.sqrt(d['filled']), 0)
X = m - (np.cumsum(imp) - imp)
dx = np.diff(X)
big = np.where(np.abs(dx) > 0.3)[0]
# group consecutive big steps within 20ms and same sign
groups = []
cur = None
for i in big:
    s = np.sign(dx[i])
    if cur and t[i] - t[cur[-1]] < 0.02 and s == cur[0]:
        cur[1].append(i); cur[-1] = i
    else:
        if cur: groups.append(cur)
        cur = [s, [i], i]
if cur: groups.append(cur)
rows = []
for s, ii, last in groups:
    a = ii[0]; b = ii[-1] + 1
    # size: X 30ms after - X 30ms before
    ia = np.searchsorted(t, t[a] - 0.03); ib = np.searchsorted(t, t[b] + 0.03)
    rows.append(dict(t=t[a], t_end=t[b], size=X[min(ib, len(X) - 1)] - X[max(ia, 0)], step=dx[ii].sum()))
J = pd.DataFrame(rows)
J = J[np.abs(J['size']) > thr].reset_index(drop=True)
print('n jumps |size|>%.1f:' % thr, len(J), 'rate/s', len(J) / 6000)
print('size quantiles', np.percentile(np.abs(J['size']), [5, 25, 50, 75, 95, 99]).round(2), 'max', np.abs(J['size']).max())
print('sign mean', np.sign(J['size']).mean(), 'mean size', J['size'].mean())
sg = np.sign(J['size'].to_numpy())
print('consecutive sign corr', np.corrcoef(sg[:-1], sg[1:])[0, 1])
print('duration (t_end - t) pct ms', (np.percentile(J.t_end - J.t, [50, 90, 99]) * 1e3).round(1))
gaps = np.diff(J.t.to_numpy())
print('gap CV', gaps.std() / gaps.mean(), 'gap pct', np.percentile(gaps, [10, 50, 90]).round(2))
print('per 300s counts', np.bincount((J.t // 300).astype(int), minlength=20))
print('per 300s net', pd.Series(J['size'].to_numpy()).groupby((J.t // 300).astype(int).to_numpy()).sum().round(0).to_dict())
J.to_csv(prep.common.CACHE / f'jumps_{tag}.csv', index=False)
h = np.histogram(np.abs(J['size']), bins=[1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 128])
print(list(zip(h[1][:-1], h[0])))
