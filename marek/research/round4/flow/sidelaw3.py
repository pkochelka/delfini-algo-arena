"""Two-class side law: probit sigma for regular-limit and tail-limit takers by regime; TWAP episode side vs S-mid at
first child; r3 control (same law on round-3 final S)."""
import numpy as np, pandas as pd
from pathlib import Path
from scipy.optimize import minimize
from scipy.stats import norm
D = Path(__file__).parent; C = D / 'cache'
def probit(x, y):
    f = lambda b: -np.sum(norm.logcdf((2 * y - 1) * (b[0] + b[1] * x)))
    res = minimize(f, [0, .2], method='Nelder-Mead'); p0 = y.mean()
    return 1 / res.x[1], res.x[0], -res.fun - len(y) * (-(p0 * np.log(p0) + (1 - p0) * np.log(1 - p0)))
tk = pd.read_parquet(C / 'takers_s.parquet')
S = tk.s_hat.to_numpy(); s = tk.sg.to_numpy(); y = (s > 0).astype(float); reg = tk.reg.to_numpy()
x = S - tk.mid.to_numpy(); k = np.where(reg == 1, 2.55, 1.0); tail = s * (tk.px.to_numpy() - S) - k > 0.5
ok = ((~tk.tw) & ((tk.shock_dt < -1) | (tk.shock_dt > 8)) & (tk.s_sd < 0.3)).to_numpy() & np.isfinite(x) & (tk.t > 25).to_numpy()
for R in [0, 1]:
    for nm, c in [('regular', ~tail), ('tail', tail)]:
        m = ok & (reg == R) & c; sg_, b0, dll = probit(x[m], y[m]); print(f'reg {R} {nm:8s} n {m.sum():6d} sigma {sg_:5.2f} icpt {b0:+.3f} dLL {dll:.1f}')
# TWAP: first child of each chain
tw = tk[tk.tw].groupby('tw_cid').head(1)
xt = (tw.s_hat - tw.mid).to_numpy(); print('TWAP first child: n', len(tw), 'corr(side, S-mid)', np.corrcoef(tw.sg, xt)[0, 1].round(3),
      'P(side == sign(S-mid))', np.mean(np.sign(xt) == tw.sg).round(3))
# r3 control
R3 = D.parents[1] / 'round3'
try:
    sp = pd.read_parquet(R3 / 's_dynamics/s_path.parquet'); r = pd.read_pickle(R3 / 'flow/cache/rows.pkl')
    ins = r[r.op == 0].reset_index(drop=True); m3 = ins.ioc.to_numpy()
    t3 = ins[m3]; S3 = sp.s_hat.to_numpy()[m3]; x3 = S3 - ((t3.bid_before + t3.ask_before) / 2).to_numpy()
    y3 = (t3.side.to_numpy() == 0).astype(float); v3 = sp.vol.to_numpy()[m3]; ok3 = np.isfinite(x3) & (sp.s_sd.to_numpy()[m3] < 0.3)
    for R in [0, 1]:
        q = ok3 & (v3 == R); sg_, b0, dll = probit(x3[q], y3[q]); print(f'r3 reg {R} n {q.sum()} sigma {sg_:.1f} (slope {1 / sg_:+.4f}) dLL {dll:.1f}')
except Exception as e: print('r3 control failed', e)
