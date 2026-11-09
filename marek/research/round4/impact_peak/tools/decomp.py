"""Decompose the round-4 post-taker mid response into: side-law gap at trade time A = sg(S0 - m0), S move B(h) = sg(S(t+h) - S0),
book gap C(h) = sg(mid(t+h) - S(t+h)). Impact(h) = A + B + C. Split by regime / class / size; exclude shock windows.
uv run --with pyarrow python marek/research/round4/impact_peak/tools/decomp.py"""
import numpy as np, pandas as pd, pathlib
R4 = pathlib.Path(__file__).resolve().parents[2]
tk = pd.read_parquet(R4 / 'flow/cache/takers_s.parquet')
rows = pd.read_pickle(R4 / 'flow/cache/rows.pkl')
sp = pd.read_parquet(R4 / 's_dynamics/s_path.parquet')
rt = rows.t.values; rmid = 0.5 * (rows.bid_after.values + rows.ask_after.values)
st = sp.t_us.values * 1e-6; ss = sp.s_hat.values
H = [0.005, 0.1, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30, 60]
def at(tq, T, X):
    i = np.searchsorted(T, tq, side='right') - 1
    return X[np.clip(i, 0, len(X) - 1)]
t = tk.t.values; sg = tk.sg.values.astype(float); m0 = tk.mid.values; S0 = tk.s_hat.values
ok = np.isfinite(m0) & (t < 6000 - 60)
on = pd.read_csv(R4 / 'flow/shock_events.csv').onset.values
dsh = np.min(np.abs(t[:, None] - on[None, :] - 8), axis=1)  # near window onset-1..onset+20 (+60 s horizon)
nshock = np.ones(len(t), bool)
for o in on:
    nshock &= ~((t > o - 61) & (t < o + 20))
out = {}
for h in H:
    mh = at(t + h, rt, rmid); Sh = at(t + h, st, ss)
    out[h] = (sg * (mh - m0), sg * (S0 - m0) + 0 * h, sg * (Sh - S0), sg * (mh - Sh))
def table(mask, name):
    print(f'\n## {name}  n={mask.sum()}')
    print('h      impact   A=gap0   B=dS     C=mid-S')
    for h in H:
        I, A, B, C = (x[mask] for x in out[h])
        f = np.isfinite(I) & np.isfinite(C)
        print(f'{h:6} {I[f].mean():7.3f} {A[f].mean():7.3f} {B[f].mean():7.3f} {C[f].mean():7.3f}')
vol = tk.vol1s.values.astype(bool) if 'vol1s' in tk else tk.svol.values
svol = tk.svol.values.astype(bool)
tw = tk.tw.values.astype(bool)
filled = tk.filled.values
base = ok & (filled > 0)
table(base, 'all filled takers (incl shocks)')
table(base & nshock, 'filled, no shock')
table(base & nshock & ~svol, 'calm (s_path vol label), no shock')
table(base & nshock & svol, 'volatile, no shock')
table(base & nshock & tw, 'TWAP children, no shock')
table(base & nshock & ~tw & svol, 'regular volatile no shock')
table(base & nshock & ~tw & ~svol, 'regular calm no shock')
table(base & ~nshock, 'shock windows')
for lo, hi in [(1, 20), (20, 80), (80, 300), (300, 10**9)]:
    table(base & nshock & ~tw & svol & (filled >= lo) & (filled < hi), f'regular vol filled {lo}-{hi}')
np.savez('/tmp/_none.npz') if False else None
