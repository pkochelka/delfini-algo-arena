"""Stability of the post-peak fade beyond the filter: anchored at A s after each filled regular volatile taker,
mean sg(S(t+A+h) - Fpred_A(h)) per 500 s window (summed over takers in the window / n), h = 10 s."""
import sys, pathlib, numpy as np, pandas as pd
R4 = pathlib.Path(__file__).resolve().parents[2]; HERE = pathlib.Path(__file__).resolve().parents[1]
z = np.load(HERE / 'cache/r4_best_round4.npz')['x']
gt, gF, gsd, gpv, gkv, gY, gturb, gpos, gbb, gba, gshk = z.T
sp = pd.read_parquet(R4 / 's_dynamics/s_path.parquet'); st = sp.t_us.values * 1e-6; ss = sp.s_hat.values
tk = pd.read_parquet(R4 / 'flow/cache/takers_s.parquet'); tk = tk[(tk.filled > 0) & ~tk.tw]
on = pd.read_csv(R4 / 'flow/shock_events.csv').onset.values
t = tk.t.values; sg = tk.sg.values.astype(float)
ok = (t > 30) & (t < 5940)
for o in on:
    ok &= ~((t > o - 61) & (t < o + 20))
at = lambda q, T, X: X[np.clip(np.searchsorted(T, q, 'right') - 1, 0, len(X) - 1)]
i0 = np.clip(np.searchsorted(gt, t, 'right'), 0, len(gt) - 1); m = ok & (gturb[i0] > 0.5)
for A, h in ((5, 10), (5, 20), (10, 10)):
    iA = np.clip(np.searchsorted(gt, t + A, 'right') - 1, 0, len(gt) - 1)
    pv = gpv[iA]; tau = (1 - pv) * 0.5 + pv * 2.1; T = (1 - pv) * 16.7 + pv * 15.0
    Fp = gF[iA] + gkv[iA] * tau * (1 - np.exp(-h / tau)) - gY[iA] * (1 - np.exp(-h / T))
    r = sg * (at(t + A + h, st, ss) - Fp)
    w = (t // 500).astype(int)
    per = [r[m & (w == k)].mean() if (m & (w == k)).sum() > 50 else np.nan for k in range(12)]
    print(f'A={A} h={h}: all {r[m].mean():+.2f}  windows ' + ' '.join(f'{x:+.1f}' for x in per))
