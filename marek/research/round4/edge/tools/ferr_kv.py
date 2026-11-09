"""F - S error of logged runs (logx npz) in volatile time by |kv| bucket: does the tracker lag S in strong trends?
Mean error is signed by the trend (negative = F behind S). Shock windows removed.
uv run --with pyarrow python marek/research/round4/edge/tools/ferr_kv.py cache/a.npz [cache/b.npz ...]"""
import sys, pathlib, numpy as np, pandas as pd
R4 = pathlib.Path(__file__).resolve().parents[2]
sp = pd.read_parquet(R4 / 's_dynamics/s_path.parquet', columns=['t_us', 's_hat'])
st = sp.t_us.values * 1e-6
on = pd.read_csv(R4 / 'flow/shock_events.csv').onset.values
B = ((0, 1), (1, 3), (3, 6), (6, 99))
print(f'{"file":22s} {"turb":>6s} {"calm":>6s} | vol rmse / mean*sign(kv) by |kv| ' + '  '.join(f'[{a},{b})' for a, b in B) + ' | share')
for fn in sys.argv[1:]:
    x = np.load(fn)['x']
    t, F, sd, pv, turb, pos, bb, ba, Y, kv, sgw = x.T
    i = np.clip(np.searchsorted(st, t, 'right') - 1, 0, len(st) - 1)
    S = sp.s_hat.values[i]
    sh = np.zeros(len(t), bool)
    for o in on:
        sh |= (t >= o - 5) & (t < o + 25)
    e = F - S
    mt = (turb == 1) & ~sh & (sgw == 0)
    mc = (turb == 0) & ~sh & (sgw == 0)
    cells = []
    shares = []
    for a, b in B:
        m = mt & (np.abs(kv) >= a) & (np.abs(kv) < b)
        cells.append(f'{np.sqrt(np.mean(e[m] ** 2)):.2f}/{np.mean(e[m] * np.sign(kv[m])):+.2f}' if m.sum() else '   -   ')
        shares.append(m.sum() / max(mt.sum(), 1))
    print(f'{fn.split("/")[-1][:22]:22s} {np.sqrt(np.mean(e[mt] ** 2)):6.3f} {np.sqrt(np.mean(e[mc] ** 2)):6.3f} | '
          + '  '.join(cells) + ' | ' + ' '.join(f'{s:.2f}' for s in shares))
