import numpy as np, pandas as pd, sys
import prep
tag = sys.argv[1] if len(sys.argv) > 1 else 'v1'
G = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
J = pd.read_csv(prep.common.CACHE / f'jumps_{tag}.csv')
J = J[np.abs(J['size']) < 12]
tt = J.t.to_numpy()
eps_ = []
s = tt[0]; last = tt[0]; n = 1
for x in tt[1:]:
    if x - last < G:
        last = x; n += 1
    else:
        eps_.append((s, last, n)); s = last = x; n = 1
eps_.append((s, last, n))
E = pd.DataFrame(eps_, columns=['start', 'end', 'n'])
E = E[E.n >= 8].reset_index(drop=True)
E['dur'] = E.end - E.start
E['rate'] = E.n / E.dur
d = prep.build(); r = dict(np.load(prep.common.CACHE / f'smooth_{tag}.npz'))
t = d['t']; m = r['mean']
def at(x): return m[np.clip(np.searchsorted(t, x) - 1, 0, len(t) - 1)]
E['dS'] = at(E.end) - at(E.start)
E['net_small'] = [J['size'][(J.t >= a) & (J.t <= b)].sum() for a, b in zip(E.start, E.end)]
pd.set_option('display.width', 200)
print(E.round(2).to_string())
print('total vol time', E.dur.sum(), 'share', E.dur.sum() / 6000)
gaps = E.start.to_numpy()[1:] - E.end.to_numpy()[:-1]
print('calm gaps', np.round(np.sort(gaps), 0))
print('mean dur', E.dur.mean(), 'mean calm gap', gaps.mean())
E.to_csv(prep.common.CACHE / f'episodes_{tag}.csv', index=False)
