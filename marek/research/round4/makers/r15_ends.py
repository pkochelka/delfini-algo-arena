import numpy as np, pandas as pd
from prep import load
o = load().reset_index(drop=True)
q = pd.read_pickle('cache/r15_chains.pkl')
c = q[q.L >= 2].copy()
rows = []
tk = o[o.ioc]
tkt = tk.t_us.to_numpy()
for ch, g in c.groupby('chain'):
    g = g.sort_values('k')
    f = g.iloc[0]; l = g.iloc[-1]
    j = np.searchsorted(tkt, f.t_us) - 1
    prevt = tk.iloc[j]
    touch = f.ask if f.sg < 0 else f.bid
    rows.append(dict(chain=ch, t0=f.t_us / 1e6, sg=f.sg, n=len(g), filled=g.filled.sum(), end=l.end, last_filled=l.filled,
                     dt_prev_taker_ms=(f.t_us - prevt.t_us) / 1e3, prev_taker_sg=prevt.sg, prev_taker_vol=prevt.volume,
                     start_rel_touch=(f.price - touch) * -f.sg, end_t=l.end_t_us / 1e6,
                     n_fill_end=(g.end == 'filled').sum(), n_canc=(g.end == 'cancelled').sum(),
                     partial=((g.filled > 0) & (g.filled < 15)).sum()))
r = pd.DataFrame(rows).sort_values('t0')
pd.set_option('display.width', 250)
print(r.to_string())
print(r.end.value_counts(), r.start_rel_touch.value_counts())
