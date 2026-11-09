"""Split the exact-S move after a filled regular taker, B(h) = sg(S(t+h) - S(t)), into generator parts:
own kick (permanent + decaying transient), later takers' kicks in (t, t+h], decay of the transient Y left by earlier
takers, and the rest (= drift of S, i.e. the velocity V, plus noise). Generator kick law: s_dynamics fit
(calm 0.0145 q^0.62, p .56, T 14; volatile 0.048 q^0.65, p .61, T 18). Shock windows out, regime from s_path vol label."""
import pathlib, numpy as np, pandas as pd
R4 = pathlib.Path(__file__).resolve().parents[2]
tk = pd.read_parquet(R4 / 'flow/cache/takers_s.parquet'); tk = tk[tk.filled > 0].reset_index(drop=True)
sp = pd.read_parquet(R4 / 's_dynamics/s_path.parquet'); st = sp.t_us.values * 1e-6; ss = sp.s_hat.values
on = pd.read_csv(R4 / 'flow/shock_events.csv').onset.values
t = tk.t.values; sg = tk.sg.values.astype(float); q = tk.filled.values.astype(float); v = tk.svol.values.astype(bool)
c = np.where(v, 0.048, 0.0145); e = np.where(v, 0.65, 0.62); P = np.where(v, 0.61, 0.56); T = np.where(v, 18.0, 14.0)
d = sg * c * q ** e
# true transient Y just before each taker (uses per-kick T, approximately)
Y = np.zeros(len(t)); y = 0.0; tl = 0.0
for i in range(len(t)):
    y *= np.exp(-(t[i] - tl) / T[i]); tl = t[i]; Y[i] = y; y += (1 - P[i]) * d[i]
S_at = lambda x: ss[np.clip(np.searchsorted(st, x, 'right') - 1, 0, len(ss) - 1)]
ok = (t > 30) & (t < 5940) & ~tk.tw.values
for o in on:
    ok &= ~((t > o - 61) & (t < o + 20))
H = [0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30]
res = {h: np.zeros((len(t), 5)) for h in H}
idx = np.where(ok)[0]
for h in H:
    j1 = np.searchsorted(t, t + h, 'right')
    for i in idx:
        B = sg[i] * (S_at(t[i] + h) - S_at(t[i]))  # S_at(t) = S before the taker (s_hat at its own row)
        own = sg[i] * d[i] * (P[i] + (1 - P[i]) * np.exp(-h / T[i]))
        js = slice(i + 1, j1[i])
        later = sg[i] * np.sum(d[js] * (P[js] + (1 - P[js]) * np.exp(-(t[i] + h - t[js]) / T[js])))
        ydec = -sg[i] * Y[i] * (1 - np.exp(-h / T[i]))
        res[h][i] = (B, own, later, ydec, B - own - later - ydec)
for name, m in (('volatile', ok & v), ('calm', ok & ~v)):
    print(f'\n## {name} n={m.sum()}   mean sg*Y_true at trade {np.mean(sg[m]*Y[m]):.3f}')
    print('h     B=dS    own    later   Ydecay  drift(V)+rest')
    for h in H:
        print(f'{h:<5} ' + ' '.join(f'{x:7.3f}' for x in res[h][m].mean(0)))
