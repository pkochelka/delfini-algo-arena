"""Is the side chosen by marketability? fb = a buy at round(S+k) would cross the ask, fs = a sell at round(S-k) would
cross the bid (k = 1 calm, 2.55 vol). P(buy | fb, fs) by regime; same for r3 as control."""
import numpy as np, pandas as pd, sys
from pathlib import Path
C = Path(__file__).parent / 'cache'
tk = pd.read_parquet(C / 'takers_s.parquet')
ok = ((~tk.tw) & ((tk.shock_dt < -1) | (tk.shock_dt > 8)) & (tk.s_sd < 0.3)).to_numpy() & tk.mid.notna().to_numpy()
S = tk.s_hat.to_numpy(); k = np.where(tk.reg == 1, 2.55, 1.0)
fb = tk.ask_before.to_numpy() <= np.round(S + k); fs = tk.bid_before.to_numpy() >= np.round(S - k)
s = tk.sg.to_numpy()
for R in [0, 1]:
    m = ok & (tk.reg.to_numpy() == R)
    print(f'== regime {R}')
    for a in [True, False]:
        for b in [True, False]:
            q = m & (fb == a) & (fs == b); print(f'   buy-marketable {a!s:5} sell-marketable {b!s:5}: n {q.sum():6d} ({q.sum() / m.sum():.3f})  P(buy) {np.mean(s[q] > 0):.3f}')
    print('   fill rate', np.mean(tk.filled.to_numpy()[m] > 0).round(4))
print('--- within classes: corr(sign, S-mid), and P(buy) vs margin')
x = S - tk.mid.to_numpy()
mb = np.round(S + k) - tk.ask_before.to_numpy()  # buy margin (>=0 marketable)
ms = tk.bid_before.to_numpy() - np.round(S - k)  # sell margin
for R in [0, 1]:
    m = ok & (tk.reg.to_numpy() == R)
    q = m & fb & fs; print(f'reg {R} both: corr(sign,S-mid) {np.corrcoef(s[q], x[q])[0, 1]:.3f}; P(buy) by (mb,ms):')
    tab = pd.crosstab(np.clip(mb[q], 0, 4), np.clip(ms[q], 0, 4), values=(s[q] > 0), aggfunc='mean').round(3)
    cnt = pd.crosstab(np.clip(mb[q], 0, 4), np.clip(ms[q], 0, 4))
    print(tab); print(cnt)
    q = m & fb & ~fs; print(f' only buy mkt: P(buy) by sell margin', pd.Series(s[q] > 0).groupby(np.clip(ms[q], -4, -1)).agg(['mean', 'size']).round(3).to_dict())
    q = m & ~fb & fs; print(f' only sell mkt: P(buy) by buy margin', pd.Series(s[q] > 0).groupby(np.clip(mb[q], -4, -1)).agg(['mean', 'size']).round(3).to_dict())
