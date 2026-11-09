"""Event study around filled regular takers, relative to r4_best's filter just after the trade (first 10 ms grid point).
Columns: impact sg(mid(t+h)-m0); sg(S(t+h)-F0) (filter error vs fixed F0); sg(S(t+h)-Fp(h)) (vs filter's own forecast
with kv drift and Y decay); sg(F(t+h)-F0) (filter's own path); sg(S(t+h)-F(t+h)) (live filter error).
uv run --with pyarrow python marek/research/round4/impact_peak/tools/evt.py [cache.npz]"""
import sys, pathlib, math, numpy as np, pandas as pd
R4 = pathlib.Path(__file__).resolve().parents[2]; HERE = pathlib.Path(__file__).resolve().parents[1]
z = np.load(sys.argv[1] if len(sys.argv) > 1 else HERE / 'cache/r4_best_round4.npz')['x']
gt, gF, gsd, gpv, gkv, gY, gturb, gpos, gbb, gba, gshk = z.T
sp = pd.read_parquet(R4 / 's_dynamics/s_path.parquet'); st = sp.t_us.values * 1e-6; ss = sp.s_hat.values
rows = pd.read_pickle(R4 / 'flow/cache/rows.pkl'); rt = rows.t.values; rmid = 0.5 * (rows.bid_after.values + rows.ask_after.values)
tk = pd.read_parquet(R4 / 'flow/cache/takers_s.parquet'); tk = tk[(tk.filled > 0) & ~tk.tw]
on = pd.read_csv(R4 / 'flow/shock_events.csv').onset.values
t = tk.t.values; sg = tk.sg.values.astype(float); m0 = tk.mid.values
ok = (t > 30) & (t < 5940)
for o in on:
    ok &= ~((t > o - 61) & (t < o + 20))
def at(q, T, X):
    return X[np.clip(np.searchsorted(T, q, 'right') - 1, 0, len(X) - 1)]
i0 = np.clip(np.searchsorted(gt, t, 'right'), 0, len(gt) - 1)  # first grid point after the trade
F0, kv0, Y0, pv0, tb0 = gF[i0], gkv[i0], gY[i0], gpv[i0], gturb[i0]
tau = (1 - pv0) * 0.5 + pv0 * 2.1; T = (1 - pv0) * 16.7 + pv0 * 15.0
H = [0.1, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30]
for name, m in (('volatile (turb)', ok & (tb0 > 0.5)), ('calm', ok & (tb0 < 0.5))):
    print(f'\n## {name}, regular filled takers n={m.sum()}')
    print('h     impact  S-F0   S-Fpred  F-F0   S-F(t+h)  [5s-30s fade vs F0 in ticks]')
    for h in H:
        Sh = at(t + h, st, ss); mh = at(t + h, rt, rmid); ih = np.clip(np.searchsorted(gt, t + h, 'right') - 1, 0, len(gt) - 1)
        Fp = F0 + kv0 * tau * (1 - np.exp(-h / tau)) - Y0 * (1 - np.exp(-h / T))
        c = [sg * (mh - m0), sg * (Sh - F0), sg * (Sh - Fp), sg * (gF[ih] - F0), sg * (Sh - gF[ih])]
        print(f'{h:<5} ' + ' '.join(f'{x[m].mean():7.3f}' for x in c))
