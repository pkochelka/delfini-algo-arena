"""Build a 50 ms grid for round 4: mid, S (s_hat), KF filtered state (F, Y, V), regime, shock mask -> ../cache/grid.npz"""
import pathlib, numpy as np, pandas as pd
H = pathlib.Path(__file__).resolve().parents[1]; R4 = H.parent
C = H / 'cache'; C.mkdir(exist_ok=True)
dt = 0.05; g = np.arange(0, 6000, dt) + dt  # value at end of each bin (last observation <= g)
r = pd.read_pickle(R4 / 'flow/cache/rows.pkl')
rt = r.t.to_numpy(); mid = ((r.bid_after + r.ask_after) / 2).ffill().bfill().to_numpy()
i = np.clip(np.searchsorted(rt, g, 'right') - 1, 0, len(rt) - 1); M = mid[i]
sp = pd.read_parquet(R4 / 's_dynamics/s_path.parquet')
st = sp.t_us.to_numpy() / 1e6
j = np.clip(np.searchsorted(st, g, 'right') - 1, 0, len(st) - 1)
S = sp.s_hat.to_numpy()[j]; SF = sp.s_filt.to_numpy()[j]; vol = sp.vol.to_numpy()[j]
z = np.load(R4 / 's_dynamics/_cache/kfd_v2b_D_state.npz')
k = np.clip(np.searchsorted(z['t'], g, 'right') - 1, 0, len(z['t']) - 1)
KF = z['filt'][k]; KS = z['smooth'][k]
on = pd.read_csv(R4 / 'flow/shock_events.csv').onset.to_numpy()
shock = np.zeros(len(g), bool)
for a in on: shock[(g >= a - 1) & (g < a + 20)] = True
# 1-s regime labels too (eval4's)
reg = pd.read_csv(R4 / 'flow/regimes_1s.csv')
v1 = np.zeros(len(g), bool)
for a, b, x in zip(reg.start, reg.end, reg.regime):
    if x == 'volatile': v1[(g >= a) & (g < b)] = True
np.savez(C / 'grid.npz', g=g, M=M, S=S, SF=SF, vol=vol, KF=KF, KS=KS, shock=shock, vol1=v1, on=on)
print(len(g), vol.mean(), shock.mean())
