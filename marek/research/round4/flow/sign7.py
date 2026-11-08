"""P(buy | S - mid) shape by regime; robustness using causal s_filt and S from the previous insert (excludes own obs)."""
import numpy as np, pandas as pd
from pathlib import Path
C = Path(__file__).parent / 'cache'
tk = pd.read_parquet(C / 'takers_s.parquet'); z = np.load(C / 's_series.npz')
r = pd.read_pickle(C / 'rows.pkl'); ins = r[r.op == 0].reset_index(drop=True); itk = np.flatnonzero(ins.ioc.to_numpy())
s_prev = z['s'][np.maximum(itk - 1, 0)]; sf_prev = z['sf'][np.maximum(itk - 1, 0)]
ok = ((~tk.tw) & ((tk.shock_dt < -1) | (tk.shock_dt > 8)) & (tk.s_sd < 0.5)).to_numpy() & tk.mid.notna().to_numpy() & (tk.t > 25).to_numpy()
s = tk.sg.to_numpy(); mid = tk.mid.to_numpy()
for R in [0, 1]:
    m = ok & (tk.reg.to_numpy() == R)
    print(f'== regime {R}')
    for nm, S in [('s_hat', tk.s_hat.to_numpy()), ('s_filt', tk.s_filt.to_numpy()), ('s_hat prev insert', s_prev), ('s_filt prev insert', sf_prev)]:
        x = S - mid; print(f'   corr(sign, {nm} - mid) = {np.corrcoef(s[m], x[m])[0, 1]:.3f}')
    x = (tk.s_hat.to_numpy() - mid)
    b = np.r_[-np.inf, np.arange(-3, 3.01, 0.25), np.inf]; k = np.digitize(x, b) - 1
    print('   bin(S-mid) P(buy) n:')
    for j in range(len(b) - 1):
        q = m & (k == j)
        if q.sum() > 30: print(f'     [{b[j]:5.2f},{b[j + 1]:5.2f})  {np.mean(s[q] > 0):.3f}  {q.sum()}')
    print('   sd(S-mid)', np.std(x[m]).round(3))
