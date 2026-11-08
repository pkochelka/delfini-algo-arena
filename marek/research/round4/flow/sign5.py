"""What carries the persistence? Volatile & calm, non-TWAP ex-shock targets. Predict sign from the previous taker split
by (filled>0 vs unfilled), by size, by limit aggressiveness; and saturation vs count imbalance in the last 1 s."""
import numpy as np, pandas as pd
from pathlib import Path
C = Path(__file__).parent / 'cache'
tk = pd.read_parquet(C / 'takers.parquet')
t = tk.t.to_numpy(); s = tk.sg.to_numpy(); f = tk.filled.to_numpy(); v = tk.vol.to_numpy(); reg = tk.reg.to_numpy()
ok = ((~tk.tw) & ((tk.shock_dt < -1) | (tk.shock_dt > 8))).to_numpy()
i = np.arange(1, len(t)); p = i - 1
for R in [0, 1]:
    m = ok[i] & ok[p] & (reg[i] == R) & (reg[p] == R) & (t[i] - t[p] < 0.5)
    pr = s[i] * s[p]
    print(f'== regime {R}: corr(s_i, s_prev) by prev type')
    for nm, mm in [('prev filled', f[p] > 0), ('prev unfilled', f[p] == 0), ('prev full fill', f[p] == v[p]), ('prev partial', (f[p] > 0) & (f[p] < v[p])),
                   ('prev vol<30', v[p] < 30), ('prev vol 30-150', (v[p] >= 30) & (v[p] < 150)), ('prev vol>=150', v[p] >= 150)]:
        q = m & mm; print(f'   {nm:16s} n={q.sum():6d} corr {pr[q].mean():+.3f} ± {pr[q].std() / np.sqrt(q.sum()):.3f}')
    # saturation: buy share vs count imbalance in last 1 s (non-TWAP takers only)
    tt = t[ok]; ss = s[ok]; rr = reg[ok]
    cs = np.r_[0, np.cumsum(ss)]; lo = np.searchsorted(tt, tt - 1.0); imb = cs[np.arange(len(tt))] - cs[lo]
    q = (rr == R) & (tt > 2)
    print('   imb(last 1 s) -> P(buy):', {k: (round(((ss[q & (imb == k)] + 1) / 2).mean(), 3), int((q & (imb == k)).sum())) for k in range(-6, 7) if (q & (imb == k)).sum() > 50})
