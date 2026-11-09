"""Offline fit of the lean: S(t+h) - F(t) on [1, Xf, Xs] alone and with the filter base [kv, Y, F-mid]; Xf/Xs = decayed
signed kick flow (filter kick law: c_v 0.05 q^0.64 vol, 0.0168 q^0.616 calm) with taus tf/ts. 100 ms grid, shocks out."""
import pathlib, numpy as np, pandas as pd
R4 = pathlib.Path(__file__).resolve().parents[2]; HERE = pathlib.Path(__file__).resolve().parents[1]
z = np.load(HERE / 'cache/r4_best_round4.npz')['x'][::10]
t, F, sd, pv, kv, Y, turb, pos, bb, ba, shk = z.T
sp = pd.read_parquet(R4 / 's_dynamics/s_path.parquet'); st = sp.t_us.values * 1e-6; ss = sp.s_hat.values
S_at = lambda q: ss[np.clip(np.searchsorted(st, q, 'right') - 1, 0, len(ss) - 1)]
tk = pd.read_parquet(R4 / 'flow/cache/takers_s.parquet'); tk = tk[tk.filled > 0]
tt = tk.t.values; q = tk.filled.values.astype(float); v = tk.svol.values
d = tk.sg.values * np.where(v, 0.05 * q ** 0.64, 0.0168 * q ** 0.616)
def X(tau):  # decayed flow at grid times
    out = np.zeros(len(t)); x = 0.0; tl = 0.0; j = 0
    for i, ti in enumerate(t):
        while j < len(tt) and tt[j] <= ti:
            x *= np.exp(-(tt[j] - tl) / tau); tl = tt[j]; x += d[j]; j += 1
        out[i] = x * np.exp(-(ti - tl) / tau)
    return out
on = pd.read_csv(R4 / 'flow/shock_events.csv').onset.values
ok = (t > 30) & (t < 5960) & (shk == 0)
for o in on:
    ok &= ~((t > o - 31) & (t < o + 50))
mid = 0.5 * (bb + ba)
half = t < 3000
for tf, ts in ((1.5, 12), (2.0, 20), (1.0, 8)):
    Xf, Xs = X(tf), X(ts)
    for name, m0 in (('vol', ok & (turb > 0.5)), ('calm', ok & (turb < 0.5))):
        print(f'\n## tf {tf} ts {ts} {name}: sd Xf {Xf[m0].std():.2f} Xs {Xs[m0].std():.2f} corr(Xs,Y) {np.corrcoef(Xs[m0], Y[m0])[0,1]:.2f}')
        for h in (0.5, 1, 2, 5, 10):
            y = S_at(t + h) - F
            out = []
            for cols in ([Xf, Xs], [kv, Y, F - mid, Xf, Xs]):
                A = np.stack([np.ones_like(t)] + cols, 1)
                bs = [np.linalg.lstsq(A[m], y[m], rcond=None)[0] for m in (m0, m0 & half, m0 & ~half)]
                out.append(' '.join(f'{bs[0][k]:+.3f}[{bs[1][k]:+.2f},{bs[2][k]:+.2f}]' for k in (-2, -1)))
            print(f'h {h:<4} alone Xf,Xs: {out[0]}   | +base Xf,Xs: {out[1]}')
