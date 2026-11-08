"""Calm: drift of exact S vs the transient impact stock Y (tape fills). Upper bound of an inventory target."""
import csv, pathlib, numpy as np, pandas as pd
R3 = pathlib.Path(__file__).resolve().parents[2]
tr = pd.read_pickle(R3 / 'flow/cache/trades.pkl')
q = tr.groupby('taker_row').agg(t_us=('t_us', 'first'), Q=('qty', 'sum'), ag=('aggressor', 'first')).sort_values('t_us')
sp = pd.read_parquet(R3 / 's_dynamics/s_path.parquet', columns=['t_us', 's_hat'])
ST, SS = sp.t_us.values, sp.s_hat.values
def S(t):
    i = np.searchsorted(ST, t, side='right') - 1
    return SS[np.clip(i, 0, len(SS) - 1)]
sg = np.where(q.ag.values == 'A', -1.0, 1.0)
jump = S(q.t_us.values + 3000) - S(q.t_us.values - 1)
print('sign check corr(sign, dS at taker)', np.corrcoef(sg, jump)[0, 1])
vs = [(float(r['start']), float(r['end'])) for r in csv.DictReader(open(R3 / 's_dynamics/vol_segments.csv'))]
on = [float(r['onset']) for r in csv.DictReader(open(R3 / 'flow/shock_events.csv'))]
tq = q.t_us.values * 1e-6
vol = np.zeros(len(tq), bool)
for a, b in vs: vol |= (tq >= a) & (tq < b)
c, e, p, T = np.where(vol, 0.062, 0.0168), np.where(vol, 0.63, 0.616), np.where(vol, 0.55, 0.50), np.where(vol, 13.6, 16.7)
kick = sg * c * q.Q.values ** e * (1 - p)
# Y on a 0.1 s grid (single T 16.7 for simplicity in calm)
dt = 0.1; G = np.arange(0, tq.max() + 1, dt)
idx = np.searchsorted(G, tq)
add = np.bincount(idx, weights=kick, minlength=len(G))[:len(G)]
Y = np.zeros(len(G)); dec = np.exp(-dt / 16.7); y = 0.0
for i in range(len(G)):
    y = y * dec + add[i]; Y[i] = y
g = np.zeros(len(G), int)
for a, b in vs: g[(G >= a) & (G < b)] = 1
for o in on: g[(G >= o - 1) & (G < o + 25)] = 2
Sg = S((G * 1e6).astype(np.int64))
print('calm sd(Y)', Y[g == 0].std(), ' vol sd(Y)', Y[g == 1].std())
for h in (1, 5, 10, 30):
    k = int(h / dt); m = (g[:-k] == 0) & (g[k:] == 0)
    dS = (Sg[k:] - Sg[:-k])[m]; y = Y[:-k][m]
    b = np.polyfit(y, dS, 1)[0]
    print(f"h={h:3d}s beta(dS on Y)={b:+.3f} (model -(1-e^-h/T)={-(1 - np.exp(-h / 16.7)):+.3f}) corr={np.corrcoef(y, dS)[0, 1]:+.3f} "
          f"E|Y|*|beta|={np.mean(np.abs(y)) * abs(b):.3f}")
# ideal inventory-target value: pos = 100*sign(-Y) held through calm, gain per s
m = (g[:-10] == 0) & (g[10:] == 0)
dS1 = (Sg[10:] - Sg[:-10])
for thr in (0.0, 0.2, 0.5, 1.0):
    pos = np.where(np.abs(Y[:-10]) > thr, -100 * np.sign(Y[:-10]), 0)
    print(f"oracle target |Y|>{thr}: share {np.mean(np.abs(Y[:-10][m]) > thr):.2f}  gain {np.sum((pos * dS1)[m][::10]):9.0f} over {m.sum() * dt:.0f} s")
np.save(R3 / 'ideas/calm/Y_grid.npy', np.c_[G, Y, g])
