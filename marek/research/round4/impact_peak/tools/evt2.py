"""Anchor at t+A after a filled regular taker (A = 2, 5, 10 s): does S revert over the next h beyond the filter's
forecast from its state at t+A?  Prints sg(S(t+A+h) - F(t+A)), sg(S - Fpred_A(h)), mean sg*kv_A, sg*Y_A, sg*(F_A - mid_A).
uv run --with pyarrow python marek/research/round4/impact_peak/tools/evt2.py"""
import sys, pathlib, numpy as np, pandas as pd
R4 = pathlib.Path(__file__).resolve().parents[2]; HERE = pathlib.Path(__file__).resolve().parents[1]
z = np.load(sys.argv[1] if len(sys.argv) > 1 else HERE / 'cache/r4_best_round4.npz')['x']
gt, gF, gsd, gpv, gkv, gY, gturb, gpos, gbb, gba, gshk = z.T
gmid = 0.5 * (gbb + gba)
sp = pd.read_parquet(R4 / 's_dynamics/s_path.parquet'); st = sp.t_us.values * 1e-6; ss = sp.s_hat.values
tk = pd.read_parquet(R4 / 'flow/cache/takers_s.parquet'); tk = tk[(tk.filled > 0) & ~tk.tw]
on = pd.read_csv(R4 / 'flow/shock_events.csv').onset.values
t = tk.t.values; sg = tk.sg.values.astype(float)
ok = (t > 30) & (t < 5940)
for o in on:
    ok &= ~((t > o - 61) & (t < o + 20))
def at(q, T, X):
    return X[np.clip(np.searchsorted(T, q, 'right') - 1, 0, len(X) - 1)]
i0 = np.clip(np.searchsorted(gt, t, 'right'), 0, len(gt) - 1)
vol0 = gturb[i0] > 0.5
for A in (2, 5, 10):
    iA = np.clip(np.searchsorted(gt, t + A, 'right') - 1, 0, len(gt) - 1)
    FA, kvA, YA, pvA = gF[iA], gkv[iA], gY[iA], gpv[iA]
    tau = (1 - pvA) * 0.5 + pvA * 2.1; T = (1 - pvA) * 16.7 + pvA * 15.0
    for name, m in (('vol', ok & vol0), ('calm', ok & ~vol0)):
        print(f'\n## anchor A={A}s, {name}: n={m.sum()}  sg*kv={np.mean(sg[m]*kvA[m]):.3f} sg*Y={np.mean(sg[m]*YA[m]):.3f} '
              f'sg*(F-mid)={np.mean(sg[m]*(FA-gmid[iA])[m]):.3f}')
        print('h    S-F_A  S-Fpred_A   (sem)')
        for h in (1, 2, 5, 10, 20):
            Sh = at(t + A + h, st, ss)
            Fp = FA + kvA * tau * (1 - np.exp(-h / tau)) - YA * (1 - np.exp(-h / T))
            a = sg * (Sh - FA); b = sg * (Sh - Fp)
            # crude sem: clusters of 1 s
            cl = (t[m] // max(h, 1)).astype(int); s = pd.Series(b[m]).groupby(cl).sum()
            sem = np.sqrt((s ** 2).sum()) / m.sum()
            print(f'{h:<4} {a[m].mean():6.3f} {b[m].mean():6.3f}   ({sem:.3f})')
