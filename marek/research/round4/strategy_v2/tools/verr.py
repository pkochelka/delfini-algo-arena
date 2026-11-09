"""Filter error of a logged run (logf/j15log npz with kv) vs s_path S and s_dynamics' smoothed V (kfd_v2b_D_state).
By regime, by time since regime switch (vol_segments.csv), after shock jumps, and by |V|."""
import sys, numpy as np, pandas as pd, pathlib
R4 = pathlib.Path(__file__).resolve().parents[2]
z = np.load(sys.argv[1]); x = z['x']
t, F, sd, pv, turb, pos, bb, ba, kv = x.T
sp = pd.read_parquet(R4 / 's_dynamics/s_path.parquet', columns=['t_us', 's_hat'])
st = sp.t_us.values * 1e-6; sh = sp.s_hat.values
def S_at(tt): return sh[np.clip(np.searchsorted(st, tt, side='right') - 1, 0, len(sh) - 1)]
k = np.load(R4 / 's_dynamics/_cache/kfd_v2b_D_state.npz')
kt = k['t'] * (1e-6 if k['t'].max() > 1e5 else 1); Vs = k['smooth'][:, 2]
print('kfd t range', kt[0], kt[-1], 'V sd', Vs.std())
V = Vs[np.clip(np.searchsorted(kt, t, side='right') - 1, 0, len(Vs) - 1)]
S = S_at(t); e = F - S
seg = pd.read_csv(R4 / 's_dynamics/vol_segments.csv'); print(seg.columns.tolist(), len(seg))
shk = pd.read_csv(R4 / 'flow/shock_events.csv'); print(shk.columns.tolist())
c0, c1 = seg.columns[:2]
vol = np.zeros(len(t), bool)
for a, b in seg[[c0, c1]].values: vol |= (t >= a) & (t < b)
sh_ = np.zeros(len(t), bool)
for o in shk.onset.values: sh_ |= (t >= o - 1) & (t < o + 20)
def rep(nm, m):
    if m.sum() < 10: return
    fut1 = S_at(t[m] + 1.0) - F[m]
    print(f'{nm:26s} n {m.sum():7d} rmse(F-S) {np.sqrt(np.mean(e[m]**2)):.3f} bias {np.mean(e[m]):+.3f} | V sd {V[m].std():.2f} kv sd {kv[m].std():.2f} '
          f'corr {np.corrcoef(V[m], kv[m])[0,1]:+.2f} rmse(kv-V) {np.sqrt(np.mean((kv[m]-V[m])**2)):.2f} slope(V~kv) {np.polyfit(kv[m], V[m], 1)[0]:.2f} | pv {pv[m].mean():.2f} turb {turb[m].mean():.2f} pinned {np.mean(np.abs(pos[m])>=95):.2f}')
rep('calm', ~vol & ~sh_); rep('vol', vol & ~sh_); rep('shock', sh_)
print('--- by time since vol start (s)')
for lo, hi in [(-3, -1), (-1, 0), (0, 0.5), (0.5, 1), (1, 2), (2, 4), (4, 8), (8, 20), (20, 1e9)]:
    m = np.zeros(len(t), bool)
    for a, b in seg[[c0, c1]].values: m |= (t >= a + lo) & (t < min(a + hi, b if hi > 0 else a + hi))
    rep(f'start+[{lo},{hi})', m & ~sh_)
print('--- by time since vol end (s)')
for lo, hi in [(-3, -1), (-1, 0), (0, 0.5), (0.5, 1), (1, 2), (2, 5), (5, 10)]:
    m = np.zeros(len(t), bool)
    for a, b in seg[[c0, c1]].values: m |= (t >= b + lo) & (t < b + hi)
    rep(f'end+[{lo},{hi})', m & ~sh_)
print('--- vol, by |V| (smoothed)')
for lo, hi in [(0, 1), (1, 2), (2, 3), (3, 5), (5, 99)]:
    rep(f'|V| [{lo},{hi})', vol & ~sh_ & (np.abs(V) >= lo) & (np.abs(V) < hi))
print('--- vol, by sign agreement of pos and V')
m = vol & ~sh_ & (np.abs(pos) >= 50)
for nm, mm in [('pos with V', m & (np.sign(pos) == np.sign(V))), ('pos against V', m & (np.sign(pos) != np.sign(V)))]:
    rep(nm, mm)
print('--- post-jump (time after onset+3.02)')
for lo, hi in [(0, 0.5), (0.5, 1), (1, 2), (2, 5), (5, 10), (10, 17)]:
    m = np.zeros(len(t), bool)
    for o in shk.onset.values: m |= (t >= o + 3.02 + lo) & (t < o + 3.02 + hi)
    rep(f'jump+[{lo},{hi})', m)
