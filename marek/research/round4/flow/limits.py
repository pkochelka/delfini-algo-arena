"""Taker limit vs S: x = sg*(px - S). Best k (share with |x-k|<=0.5), tail share and excess, by regime; non-TWAP; TWAP separately."""
import numpy as np, pandas as pd
from pathlib import Path
C = Path(__file__).parent / 'cache'
tk = pd.read_parquet(C / 'takers_s.parquet')
x = tk.sg * (tk.px - tk.s_hat); ok = (tk.s_sd < 0.3) & ((tk.shock_dt < -1) | (tk.shock_dt > 8))
for R in [0, 1]:
    for tw in [False, True]:
        m = ok & (tk.reg == R) & (tk.tw == tw); xx = x[m].to_numpy()
        ks = np.arange(0.5, 4.01, 0.05); sh = [np.mean(np.abs(xx - k) <= 0.5) for k in ks]; kb = ks[np.argmax(sh)]
        tail = xx > kb + 0.5
        print(f'reg {R} tw {tw} n {m.sum()} best k {kb:.2f} band {max(sh):.3f} tail {tail.mean():.3f} excess mean {np.mean(xx[tail] - kb - 0.5) if tail.any() else 0:.2f} below {np.mean(xx < kb - 0.5):.3f}')
        print('    quantiles x', np.round(np.quantile(xx, [.01, .05, .25, .5, .75, .95, .99]), 2))
