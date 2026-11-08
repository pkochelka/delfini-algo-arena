"""Step 0 / H2 offline: our F (logf.py npz) vs s_dynamics s_path; error by regime / TWAP; velocity predictability."""
import sys, numpy as np, pandas as pd, pathlib
R4 = pathlib.Path(__file__).resolve().parents[2]
x = np.load(sys.argv[1])['x']
t, F, sd, pv, turb, pos, bb, ba = x.T
sp = pd.read_parquet(R4 / 's_dynamics/s_path.parquet', columns=['t_us', 's_hat'])
st = sp.t_us.values * 1e-6; sh = sp.s_hat.values
t00 = st[0] if t[0] > 1e5 else 0
def S_at(tt):
    i = np.searchsorted(st, tt, side='right') - 1
    return sh[np.clip(i, 0, len(sh) - 1)]
reg = pd.read_csv(R4 / 'flow/regimes_1s.csv'); shk = pd.read_csv(R4 / 'flow/shock_events.csv')
tw = pd.read_csv(R4 / 'novelty/twap_episodes.csv')
print('tape t0', t[0], 'spath t0', st[0], tw.columns.tolist())
rel = t - t[0]
vol = np.zeros(len(t), bool)
for a, b, g in reg[['start', 'end', 'regime']].values:
    if g == 'volatile': vol |= (rel >= a) & (rel < b)
sh_ = np.zeros(len(t), bool)
for o in shk.onset.values: sh_ |= (rel >= o - 1) & (rel < o + 20)
inw = np.zeros(len(t), bool)
c0, c1 = 't0', 't1'
for a, b in tw[[c0, c1]].values: inw |= (rel >= a) & (rel <= b)
S = S_at(t)
e = F - S
mid = 0.5 * (bb + ba)
for nm, m in [('calm', ~vol & ~sh_), ('vol', vol & ~sh_), ('calm twap', ~vol & ~sh_ & inw), ('calm notw', ~vol & ~sh_ & ~inw),
              ('vol twap', vol & ~sh_ & inw), ('vol notw', vol & ~sh_ & ~inw)]:
    print(f'{nm:10s} n {m.sum():7d} rmse {np.sqrt(np.mean(e[m]**2)):.3f} mae {np.mean(np.abs(e[m])):.3f} bias {np.mean(e[m]):+.3f}  |mid-S| {np.mean(np.abs(mid[m]-S[m])):.3f}')
# predictability: future S(t+h) - F(t) on F - EMA_tau(F), F - mid
dt = 0.01
def ema(F, tau):
    out = np.empty_like(F); a = 1 - np.exp(-dt / tau); m = F[0]
    for i, f in enumerate(F):
        m += a * (f - m); out[i] = m
    return out
half = rel < 3000
for h in (0.5, 1, 2, 5):
    y = S_at(t + h) - F
    for tau in (0.25, 0.5, 1.0, 2.0):
        g = F - ema(F, tau)
        for nm, m in [('calm', ~vol & ~sh_), ('vol', vol & ~sh_), ('turbF', (turb > 0) & ~sh_)]:
            out = []
            for hm in (half, ~half):
                mm = m & hm
                X = np.c_[g[mm], np.ones(mm.sum())]
                b, *_ = np.linalg.lstsq(X, y[mm], rcond=None)
                # OOS on the other half
                mo = m & ~hm
                r2 = 1 - np.var(y[mo] - np.c_[g[mo], np.ones(mo.sum())] @ b) / np.var(y[mo])
                out.append((b[0], r2))
            print(f'h {h:3} tau {tau:4} {nm:5s} k {out[0][0]:+.3f}/{out[1][0]:+.3f}  oosR2 {out[1][1]:+.3f}/{out[0][1]:+.3f}')
    gm = np.clip(F - mid, -3, 3)
    for nm, m in [('calm', ~vol & ~sh_), ('vol', vol & ~sh_)]:
        g1 = F - ema(F, 1.0)
        X = np.c_[g1[m], gm[m], np.ones(m.sum())]
        b, *_ = np.linalg.lstsq(X, y[m], rcond=None)
        r2 = 1 - np.var(y[m] - X @ b) / np.var(y[m])
        print(f'h {h} joint {nm}: k_vel(tau1) {b[0]:+.3f} k_mid {b[1]:+.3f} R2 {r2:.3f}')
