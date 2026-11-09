"""F - S error of logged runs (logx npz) by regime, and the share of time the take rule would be blocked by it.
uv run --with pyarrow python marek/research/round4/edge/tools/ferr2.py cache/a.npz [cache/b.npz ...]"""
import sys, pathlib, numpy as np, pandas as pd
R4 = pathlib.Path(__file__).resolve().parents[2]
sp = pd.read_parquet(R4 / 's_dynamics/s_path.parquet', columns=['t_us', 's_hat', 's_filt'])
st = sp.t_us.values * 1e-6
on = pd.read_csv(R4 / 'flow/shock_events.csv').onset.values
for fn in sys.argv[1:]:
    x = np.load(fn)['x']
    t, F, sd, pv, turb, pos, bb, ba, Y, kv, sgw = x.T
    i = np.clip(np.searchsorted(st, t, 'right') - 1, 0, len(st) - 1)
    S = sp.s_hat.values[i]; Sf = sp.s_filt.values[i]
    sh = np.zeros(len(t), bool)
    for o in on: sh |= (t >= o - 5) & (t < o + 25)
    e = F - S
    print(fn.split('/')[-1])
    for nm, m in (('turb', (turb == 1) & ~sh & (sgw == 0)), ('calm', (turb == 0) & ~sh & (sgw == 0))):
        print(f'  {nm}: F-S rmse {np.sqrt(np.mean(e[m]**2)):.3f} mean {e[m].mean():+.3f} |e|>0.5 share {np.mean(np.abs(e[m])>0.5):.3f} '
              f'|e|>1 {np.mean(np.abs(e[m])>1):.3f}; sd mean {sd[m].mean():.3f}; s_filt-S rmse {np.sqrt(np.mean((Sf-S)[m]**2)):.3f}')
