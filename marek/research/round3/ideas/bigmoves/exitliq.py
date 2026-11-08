"""Exit liquidity after non-shock stale events: new maker volume on the opposite side at >= S - u (bids, after an up
move) arriving within W ms, plus what already rests there at t."""
import sys, numpy as np, pandas as pd
from common import *
name, thr = sys.argv[1], sys.argv[2]
R = pd.read_pickle(pathlib.Path(__file__).parent / f'nonshock_{name}_thr{thr}.pkl')
X = pd.read_pickle(pathlib.Path(__file__).parent / f'nonshock_{name}.pkl'); R['d'] = X.d.values[R.n.values]
tp, o, tr = load(name); St, Sv = s_path(name)
mk = o[~o.ioc & (o.resting_quantity > 0)]
for sd in ('B', 'A'):
    m = mk[mk.side == sd]
    globals()['t_' + sd] = m.t.values; globals()['p_' + sd] = m.price.values; globals()['q_' + sd] = m.resting_quantity.values
R = R[R.vol]
out = {}
for u in (1, 2, 3):
    for W in (20, 50, 100):
        tot = []
        for t, d in zip(R.t.values, R.d.values):
            S = Sv[np.searchsorted(St, t, 'right') - 1]
            if d > 0:
                a, b = np.searchsorted(t_B, t), np.searchsorted(t_B, t + W / 1e3)
                tot.append(q_B[a:b][p_B[a:b] >= S - u].sum())
            else:
                a, b = np.searchsorted(t_A, t), np.searchsorted(t_A, t + W / 1e3)
                tot.append(q_A[a:b][p_A[a:b] <= S + u].sum())
        tot = np.array(tot)
        out[(u, W)] = (np.median(tot), tot.mean(), (tot >= 200).mean())
for k, v in out.items():
    print(f"u={k[0]} W={k[1]}ms new opp-side volume within u of S: median {v[0]:.0f} mean {v[1]:.0f} P(>=200) {v[2]:.2f}")
print('existing within u at t: dep1/dep2/dep4 mean', R.dep1.mean().round(), R.dep2.mean().round(), R.dep4.mean().round())
# time until 200 lots of new opposite-side volume within 1.5 of S, for the events with big excess value
R['Vx'] = R.V950 - R.V100
big = R[R.Vx >= 2000]
print('events with V950-V100 >= 2000:', len(big), 'sum Vx', round(big.Vx.sum()), ' all events sum Vx', round(R.Vx.sum()))
ts = []
for t, d in zip(big.t.values, big.d.values):
    S = Sv[np.searchsorted(St, t, 'right') - 1]
    if d > 0:
        a, b = np.searchsorted(t_B, t), np.searchsorted(t_B, t + 2.0); m = p_B[a:b] >= S - 1.5; tq = t_B[a:b][m]; qq = q_B[a:b][m]
    else:
        a, b = np.searchsorted(t_A, t), np.searchsorted(t_A, t + 2.0); m = p_A[a:b] <= S + 1.5; tq = t_A[a:b][m]; qq = q_A[a:b][m]
    c = np.cumsum(qq); k = np.searchsorted(c, 200)
    ts.append((tq[k] - t) * 1e3 if k < len(tq) else 2000)
print('ms until 200 lots of new exit volume within 1.5 of S: pct 10/25/50/75/90', np.percentile(ts, [10, 25, 50, 75, 90]).round())
