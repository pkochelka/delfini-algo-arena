"""Volatile/calm: regress S(t+h) - F(t) on [1, kv, Y, F-mid] (100 ms grid, shocks out); compare Y coefficient with the
filter's own decay -(1 - exp(-h/T)). Halves for stability. Also the coefficient on Y split by Y sign-agreement with kv."""
import sys, pathlib, numpy as np, pandas as pd
R4 = pathlib.Path(__file__).resolve().parents[2]; HERE = pathlib.Path(__file__).resolve().parents[1]
z = np.load(sys.argv[1] if len(sys.argv) > 1 else HERE / 'cache/r4_best_round4.npz')['x'][::10]
t, F, sd, pv, kv, Y, turb, pos, bb, ba, shk = z.T
sp = pd.read_parquet(R4 / 's_dynamics/s_path.parquet'); st = sp.t_us.values * 1e-6; ss = sp.s_hat.values
S_at = lambda q: ss[np.clip(np.searchsorted(st, q, 'right') - 1, 0, len(ss) - 1)]
on = pd.read_csv(R4 / 'flow/shock_events.csv').onset.values
ok = (t > 30) & (t < 5960) & (shk == 0)
for o in on:
    ok &= ~((t > o - 31) & (t < o + 50))
mid = 0.5 * (bb + ba)
for name, m0 in (('volatile', ok & (turb > 0.5)), ('calm', ok & (turb < 0.5))):
    T = 15.0 if name == 'volatile' else 16.7
    print(f'\n## {name}  sd(Y)={Y[m0].std():.2f} sd(kv)={kv[m0].std():.2f} sd(F-mid)={(F-mid)[m0].std():.2f}')
    print('h    model_Y   Y_all   Y_h1   Y_h2   kv_all  Fmid_all   R2')
    for h in (1, 2, 5, 10, 15, 20, 30):
        y = S_at(t + h) - F
        X = np.stack([np.ones_like(t), kv, Y, F - mid], 1)
        r = []
        for m in (m0, m0 & (t < 3000), m0 & (t >= 3000)):
            b, *_ = np.linalg.lstsq(X[m], y[m], rcond=None); r.append(b)
        e = y[m0] - X[m0] @ r[0]
        print(f'{h:<4} {-(1-np.exp(-h/T)):6.3f}  {r[0][2]:6.3f} {r[1][2]:6.3f} {r[2][2]:6.3f}  {r[0][1]:6.3f}  {r[0][3]:6.3f}  {1-e.var()/y[m0].var():.3f}')
