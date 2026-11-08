"""Upper bound for 'hold inventory deliberately on the transient-impact signal'.
x(t) = sum over tape takers of sign * c_reg * q^0.61 * exp(-(t - t_k)/T)  (the not-yet-decayed part of impact)
Regress future S change over H on x. Then compare: (a) r3_best's own inventory PnL sum inv*dS,
(b) a frictionless book that holds pos = 100*clip(-x/xs, -1, 1) (or sign) and earns pos*dS.
usage: uv run --with pyarrow python steer_bound.py RUN_LOGDIR
"""
import sys, csv, bisect, numpy as np, pandas as pd
rd = sys.argv[1]
S = pd.read_parquet('marek/research/round3/s_dynamics/s_path.parquet', columns=['t_us', 's_hat', 'vol', 'shock'])
tape = pd.read_csv('data/round3.csv', usecols=['Operation', 'Side', 'Volume', 'Lifespan'])
ins = tape[tape.Operation == 'Insert'].reset_index(drop=True)
st = S.t_us.values; sh = S.s_hat.values; vol = S.vol.values
tk = np.where(ins.Lifespan.values == 'F')[0]
sg = np.where(ins.Side.values[tk] == 'B', 1.0, -1.0); q = ins.Volume.values[tk].astype(float)
c = np.where(vol[tk], 0.062, 0.0171); kick = sg * c * q ** 0.61
grid = np.arange(0, st[-1], 100_000)
Sg = sh[np.clip(np.searchsorted(st, grid, side='right') - 1, 0, None)]
vg = vol[np.clip(np.searchsorted(st, grid, side='right') - 1, 0, None)]
R3 = 'marek/research/round3/'
on = np.array([float(r['onset']) for r in csv.DictReader(open(R3 + 'flow/shock_events.csv'))]) * 1e6
shockg = np.zeros(len(grid), bool)
for o in on: shockg |= (grid >= o - 1e6) & (grid < o + 20e6)
def xstate(T):
    # exponential filter on a 100 ms grid
    b = np.searchsorted(grid, st[tk], side='left'); add = np.bincount(b, weights=kick, minlength=len(grid))[:len(grid)]
    a = np.exp(-0.1 / T); x = np.zeros(len(grid)); acc = 0.0
    for i in range(len(grid)):
        acc = acc * a + add[i]; x[i] = acc
    return x
dS = np.diff(Sg, append=Sg[-1])
mask = ~shockg & ~np.roll(shockg, -1)
for T in (5, 15, 30):
    x = xstate(T)
    for H in (10, 50, 100, 300):   # in 100 ms steps
        fut = np.roll(Sg, -H) - Sg
        m = mask & (np.arange(len(grid)) < len(grid) - H)
        for nm, mm in (('calm', m & ~vg), ('vol', m & vg)):
            b = (x[mm] * fut[mm]).sum() / (x[mm] ** 2).sum(); r = np.corrcoef(x[mm], fut[mm])[0, 1]
            print(f'T={T} H={H/10:.0f}s {nm}: beta {b:.3f} corr {r:.3f} sd_x {x[mm].std():.2f}')
# frictionless steering PnL vs our inventory PnL
cols = ['ts', 'price', 'tside', 'qq', 'tid', 'mid', 'town', 'mown']
rt = pd.read_csv(f'{rd}/trades.csv', names=cols, header=0)
u = rt[(rt.town == 'user') | (rt.mown == 'user')]
sgn = np.where(((u.town == 'user') & (u.tside == 'B')) | ((u.mown == 'user') & (u.tside == 'A')), 1, -1)
dinv = np.bincount(np.searchsorted(grid, u.ts.values, side='left'), weights=sgn * u.qq.values, minlength=len(grid))[:len(grid)]
inv = np.cumsum(dinv)
x = xstate(15)
for nm, mm in (('calm', mask & ~vg), ('vol', mask & vg)):
    ours = (inv * dS)[mm].sum()
    sig = (-100 * np.sign(x) * dS)[mm].sum()
    print(f'{nm}: our inventory PnL {ours:.0f}; steer sign(-x)*100: {sig:.0f}; corr(inv, -x) {np.corrcoef(inv[mm], -x[mm])[0,1]:.3f}; mean|inv| {np.abs(inv[mm]).mean():.0f}')
