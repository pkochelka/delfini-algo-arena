"""Aggressive limit tail vs alignment xa = sg*(S - mid) (positive = taker on the cheap side). P(tail), excess quantiles,
and excess vs the distance to the opposite touch."""
import numpy as np, pandas as pd
from pathlib import Path
C = Path(__file__).parent / 'cache'
tk = pd.read_parquet(C / 'takers_s.parquet')
S = tk.s_hat.to_numpy(); s = tk.sg.to_numpy(); reg = tk.reg.to_numpy(); px = tk.px.to_numpy()
k = np.where(reg == 1, 2.55, 1.0); lx = s * (px - S); exc = lx - k
xa = s * (S - tk.mid.to_numpy())
far = np.where(s > 0, tk.ask_before.to_numpy() - S, S - tk.bid_before.to_numpy())  # opposite touch distance from S (taker direction)
ok = ((~tk.tw) & ((tk.shock_dt < -1) | (tk.shock_dt > 8)) & (tk.s_sd < 0.3)).to_numpy() & np.isfinite(xa) & (tk.t > 25).to_numpy()
tail = exc > 0.5
for R in [0, 1]:
    m = ok & (reg == R); print(f'== regime {R}: tail share {tail[m].mean():.3f}')
    for a, b in [(-99, -3), (-3, -1.5), (-1.5, -.5), (-.5, 0), (0, .5), (.5, 1.5), (1.5, 3), (3, 99)]:
        q = m & (xa >= a) & (xa < b)
        if q.sum() > 20:
            e = exc[q & tail]
            print(f'   xa [{a},{b}) n {q.sum():6d} P(tail) {tail[q].mean():.3f}  excess q25/50/75 {np.round(np.quantile(e, [.25, .5, .75]), 2) if len(e) > 5 else "-"}')
    # tail takers: is the limit tied to the book? limit - opposite touch
    q = m & tail; d = s[q] * (px[q] - np.where(s[q] > 0, tk.ask_before.to_numpy()[q], tk.bid_before.to_numpy()[q]))
    print('   tail: limit beyond opposite touch (ticks) value counts', pd.Series(d).value_counts().sort_index().head(15).to_dict())
    print('   tail: corr(excess, xa)', np.corrcoef(exc[q], xa[q])[0, 1].round(3), 'corr(excess, far)', np.corrcoef(exc[q], far[q])[0, 1].round(3))
    print('   tail excess quantiles', np.round(np.quantile(exc[q] - 0.5, [.1, .25, .5, .75, .9, .99]), 2), 'mean', round(np.mean(exc[q] - .5), 2))
