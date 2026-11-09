"""Forward carry E[sign(pos) * (S(t+h) - S(t))] of the r4_best position, by regime, position age and trend extent.
Usage: carry2.py LOG.npz. Tests the 'hold ~10 s then flatten' idea: if the forward carry of old / extended positions
turns negative at 5-20 s, timed or extent-based exits would pay."""
import sys, pathlib, numpy as np, pandas as pd
R4 = pathlib.Path(__file__).resolve().parents[2]
x = np.load(sys.argv[1])['x']
t, F, sd, pv, turb, pos, bb, ba = x.T
sp = pd.read_parquet(R4 / 's_dynamics/s_path.parquet', columns=['t_us', 's_hat', 'vol'])
st = sp.t_us.values * 1e-6
j = np.clip(np.searchsorted(st, t, side='right') - 1, 0, len(st) - 1)
S = sp.s_hat.values[j]; vol = sp.vol.values[j]
shk = pd.read_csv(R4 / 'flow/shock_events.csv').onset.values
sh_ = np.zeros(len(t), bool)
for o in shk: sh_ |= (t >= o - 1) & (t < o + 25)
sg = np.sign(pos)
chg = np.r_[True, sg[1:] != sg[:-1]]
last = np.maximum.accumulate(np.where(chg, np.arange(len(t)), 0))
age = t - t[last]
ext = sg * (S - S[last])  # S move in the position's favour since the sign was set
def fut(h):
    k = int(h / 0.01); return np.r_[S[k:] - S[:-k], np.full(k, np.nan)]
FU = {h: fut(h) for h in (1, 2, 5, 10, 20)}
for nm, rm in (('vol', vol & turb.astype(bool)), ('calm', ~vol)):
    m0 = rm & ~sh_ & (np.abs(pos) >= 50)
    print(f'\n=== {nm}: share of time |pos|>=50 {m0.sum()/ (rm & ~sh_).sum():.2f}')
    print('age bin      share  ' + '  '.join(f'+{h:>2}s' for h in FU))
    for a0, a1 in [(0, 2), (2, 5), (5, 10), (10, 15), (15, 20), (20, 30), (30, 60), (60, 1e9)]:
        m = m0 & (age >= a0) & (age < a1)
        print(f'{a0:>3}-{a1:<6} {m.mean()/m0.mean()*100:5.1f}%  ' + '  '.join(f'{np.nanmean(sg[m]*FU[h][m]):+5.2f}' for h in FU))
    print('extent bin (S move since sign set, ticks)')
    for a0, a1 in [(-1e9, 0), (0, 3), (3, 6), (6, 10), (10, 15), (15, 25), (25, 1e9)]:
        m = m0 & (ext >= a0) & (ext < a1)
        if m.sum() < 100: continue
        print(f'{a0:>5}-{a1:<6} {m.mean()/m0.mean()*100:5.1f}%  ' + '  '.join(f'{np.nanmean(sg[m]*FU[h][m]):+5.2f}' for h in FU))
