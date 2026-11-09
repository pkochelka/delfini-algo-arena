"""What does r4_best's own causal state predict at 1-20 s? Target S(t+h) - F(t) (S = s_path), sampled every 0.25 s.
Regressors: kv (its velocity), Y (its transient), F - mid, past F moves (2/5/10/20 s). Half-split OOS R^2."""
import sys, pathlib, numpy as np, pandas as pd
R4 = pathlib.Path(__file__).resolve().parents[2]
x = np.load(sys.argv[1])['x']
t, F, sd, pv, turb, pos, bb, ba, Y, kv = x.T
# regular 10 ms grid by forward-fill
g = np.arange(30, 5970, 0.01)
i = np.clip(np.searchsorted(t, g, 'right') - 1, 0, len(t) - 1)
F, Y, kv, turb, pos, mid = F[i], Y[i], kv[i], turb[i].astype(bool), pos[i], 0.5 * (bb[i] + ba[i])
sp = pd.read_parquet(R4 / 's_dynamics/s_path.parquet', columns=['t_us', 's_hat', 'vol'])
st = sp.t_us.values * 1e-6
S = sp.s_hat.values[np.clip(np.searchsorted(st, g, 'right') - 1, 0, len(st) - 1)]
on = pd.read_csv(R4 / 'flow/shock_events.csv').onset.values
sh = np.zeros(len(g), bool)
for o in on: sh |= (g >= o - 25) & (g < o + 25)
idx = np.arange(2000, len(g) - 2000, 25)
def lag(X, h): return X[idx] - X[idx - int(h * 100)]
half = idx < len(g) // 2
def fit(X, y, m):
    X = np.column_stack([np.ones(len(y)), X]); out = []
    for tr in (half, ~half):
        a, b = m & tr, m & ~tr
        be = np.linalg.lstsq(X[a], y[a], rcond=None)[0]; r = y[b] - X[b] @ be; out.append(1 - r.var() / y[b].var())
    return np.mean(out), np.linalg.lstsq(X[m], y[m], rcond=None)[0][1:]
G = F[idx] - mid[idx]
base = [kv[idx], Y[idx], G]
past = [lag(F, h) for h in (2, 5, 10, 20)]
for nm, rm in (('turb', turb[idx]), ('calm', ~turb[idx])):
    m = rm & ~sh[idx]
    print(f'\n== {nm} n={m.sum()}  (target S(t+h)-F(t); also R2 of S(t+h)-S(t) on the same)')
    for h in (1, 2, 5, 10, 20):
        y = S[idx + int(h * 100)] - F[idx]
        r0, b0 = fit(np.column_stack(base), y, m)
        r1, b1 = fit(np.column_stack(base + past), y, m)
        rY, bY = fit(np.column_stack([Y[idx]]), y, m)
        print(f'h={h:>2} base R2 {r0:6.3f} [kv {b0[0]:5.2f} Y {b0[1]:5.2f} F-mid {b0[2]:5.2f}] +past R2 {r1:6.3f} past b '
              + ' '.join(f'{v:5.2f}' for v in b1[3:]) + f' | Y alone R2 {rY:6.3f} b {bY[0]:5.2f}')
    print('Y sd %.2f  kv sd %.2f  F-mid sd %.2f' % (Y[idx][m].std(), kv[idx][m].std(), G[m].std()))
