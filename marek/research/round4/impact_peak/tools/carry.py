"""Forward carry of r4_best's volatile position, split by the impact-peak state: does the held position lose after the
flow peak (old flow Y backs pos, new flow Xf dead / reversed)?  E[sign(pos) (S(t+h) - S(t))] on a 100 ms grid."""
import pathlib, numpy as np, pandas as pd
R4 = pathlib.Path(__file__).resolve().parents[2]; HERE = pathlib.Path(__file__).resolve().parents[1]
z = np.load(HERE / 'cache/r4_best_round4.npz')['x'][::10]
t, F, sd, pv, kv, Y, turb, pos, bb, ba, shk = z.T
sp = pd.read_parquet(R4 / 's_dynamics/s_path.parquet'); st = sp.t_us.values * 1e-6; ss = sp.s_hat.values
S_at = lambda q: ss[np.clip(np.searchsorted(st, q, 'right') - 1, 0, len(ss) - 1)]
tk = pd.read_parquet(R4 / 'flow/cache/takers_s.parquet'); tk = tk[tk.filled > 0]
tt = tk.t.values; q = tk.filled.values.astype(float); v = tk.svol.values
d = tk.sg.values * np.where(v, 0.05 * q ** 0.64, 0.0168 * q ** 0.616)
def X(tau):
    out = np.zeros(len(t)); x = 0.0; tl = 0.0; j = 0
    for i, ti in enumerate(t):
        while j < len(tt) and tt[j] <= ti:
            x *= np.exp(-(tt[j] - tl) / tau); tl = tt[j]; x += d[j]; j += 1
        out[i] = x * np.exp(-(ti - tl) / tau)
    return out
Xf = X(1.5)
on = pd.read_csv(R4 / 'flow/shock_events.csv').onset.values
ok = (t > 30) & (t < 5960) & (shk == 0)
for o in on:
    ok &= ~((t > o - 31) & (t < o + 50))
s = np.sign(pos); m0 = ok & (turb > 0.5) & (np.abs(pos) >= 60)
yb = s * Y; xb = s * Xf
print('volatile, |pos|>=60: n', m0.sum())
print('state                         share   carry +1s   +2s   +5s  +10s  +20s (ticks of S per unit sign)')
for name, m in (('all', m0), ('Y backs pos >=3', m0 & (yb >= 3)), ('Y>=3 & Xf<=1', m0 & (yb >= 3) & (xb <= 1)),
                ('Y>=8 & Xf<=0', m0 & (yb >= 8) & (xb <= 0)), ('Y>=15 & Xf<=-2', m0 & (yb >= 15) & (xb <= -2)),
                ('Y against pos', m0 & (yb < 0)), ('Xf backs pos >=3', m0 & (xb >= 3))):
    c = [np.mean((s * (S_at(t + h) - S_at(t)))[m]) for h in (1, 2, 5, 10, 20)]
    print(f'{name:28s} {m.sum()/m0.sum():5.2f}  ' + ' '.join(f'{x:+6.2f}' for x in c))
