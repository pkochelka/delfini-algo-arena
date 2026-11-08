"""Side law details: calibration over S-mid; slope by sub-population (shock windows, spread, size, tail, time blocks);
size and limit-tail vs alignment; arrival rate vs |S-mid|."""
import numpy as np, pandas as pd
from pathlib import Path
from scipy.optimize import minimize
from scipy.stats import norm
C = Path(__file__).parent / 'cache'
tk = pd.read_parquet(C / 'takers_s.parquet')
x = (tk.s_hat - tk.mid).to_numpy(); s = tk.sg.to_numpy(); y = (s > 0).astype(float); reg = tk.reg.to_numpy()
spr = (tk.ask_before - tk.bid_before).to_numpy(); v = tk.vol.to_numpy()
k = np.where(reg == 1, 2.55, 1.0); lx = s * (tk.px.to_numpy() - tk.s_hat.to_numpy()); tail = lx > k + 0.5
base = np.isfinite(x) & (tk.s_sd < 0.5).to_numpy() & (tk.t > 25).to_numpy()
nt = ~tk.tw.to_numpy(); exs = ((tk.shock_dt < -1) | (tk.shock_dt > 8)).to_numpy()
def probit(m):
    f = lambda b: -np.sum(norm.logcdf((2 * y[m] - 1) * (b[0] + b[1] * x[m])))
    b = minimize(f, [0, .2], method='Nelder-Mead').x; return 1 / b[1], b[0], m.sum()
ok = base & nt & exs
print('calibration (all regimes): bin, n, obs P(buy), model Phi(x/4.42)')
bins = [-40, -8, -5, -3, -2, -1.5, -1, -.5, 0, .5, 1, 1.5, 2, 3, 5, 8, 40]
kk = np.digitize(x, bins) - 1
for j in range(len(bins) - 1):
    q = ok & (kk == j)
    if q.sum(): print(f'  [{bins[j]:5.1f},{bins[j + 1]:5.1f}) n {q.sum():6d} obs {y[q].mean():.3f} model {norm.cdf(x[q] / 4.42).mean():.3f}')
print('probit sigma (1/slope), intercept, n by subset:')
for nm, m in [('calm', ok & (reg == 0)), ('vol', ok & (reg == 1)), ('shock -1..8 s (non-TWAP)', base & nt & ~exs),
              ('drought only', base & nt & (tk.shock_dt.to_numpy() >= 0) & (tk.shock_dt.to_numpy() < 3)),
              ('post-jump 3..8 s', base & nt & (tk.shock_dt.to_numpy() >= 3) & (tk.shock_dt.to_numpy() < 8)),
              ('spread 1', ok & (spr == 1)), ('spread 2', ok & (spr == 2)), ('spread 3-4', ok & (spr >= 3) & (spr <= 4)), ('spread >=5', ok & (spr >= 5)),
              ('size <30', ok & (v < 30)), ('size 30-150', ok & (v >= 30) & (v < 150)), ('size >=150', ok & (v >= 150)),
              ('regular limit', ok & ~tail), ('tail limit', ok & tail),
              ('t<2000', ok & (tk.t.to_numpy() < 2000)), ('2000-4000', ok & (tk.t.to_numpy() >= 2000) & (tk.t.to_numpy() < 4000)), ('>=4000', ok & (tk.t.to_numpy() >= 4000))]:
        sg_, b0, n = probit(m); print(f'  {nm:26s} sigma {sg_:6.2f}  icpt {b0:+.3f}  n {n}')
# size & tail vs alignment (aligned = sign(x) == side), |x|>1
for R in [0, 1]:
    m = ok & (reg == R) & (np.abs(x) > 1); al = np.sign(x) == s
    print(f'reg {R}: |x|>1 aligned n {np.sum(m & al)} vs against {np.sum(m & ~al)}; mean log size {np.log(v[m & al]).mean():.3f} vs {np.log(v[m & ~al]).mean():.3f}; tail share {tail[m & al].mean():.3f} vs {tail[m & ~al].mean():.3f}')
# arrival rate vs |S - mid|: S-mid sampled at all inserts (proxy for time), taker share among inserts by |x| bin
z = np.load(C / 's_series.npz'); r = pd.read_pickle(C / 'rows.pkl'); ins = r[r.op == 0].reset_index(drop=True)
xa = z['s'] - (ins.bid_before + ins.ask_before).to_numpy() / 2; ioc = ins.ioc.to_numpy()
vv = np.zeros(len(ins), int); vs = pd.read_csv(C.parent.parent / 's_dynamics/vol_segments.csv')
ti = ins.t.to_numpy()
for a, b in zip(vs.start, vs.end): vv[(ti >= a) & (ti < b)] = 1
for R in [0, 1]:
    m = (vv == R) & np.isfinite(xa) & (ti > 25)
    print(f'reg {R} taker share of inserts by |S-mid|:', {f'{a}-{b}': round(ioc[m & (np.abs(xa) >= a) & (np.abs(xa) < b)].mean(), 4) for a, b in [(0, .5), (.5, 1), (1, 2), (2, 4), (4, 99)]})
