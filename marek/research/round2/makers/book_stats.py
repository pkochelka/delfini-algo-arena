"""Model-free book statistics for round 2: spread, touch depth, one-sided episodes, by 600 s segment."""
import numpy as np, pandas as pd
from pathlib import Path
q = pd.read_pickle(Path(__file__).parent / 'cache/quotes.pkl')
t = q.t_us.to_numpy(); dur = np.diff(np.r_[t, t[-1]]) / 1e6
bid, ask = q.bid.to_numpy(), q.ask.to_numpy()
two = ~np.isnan(bid) & ~np.isnan(ask)
sp = ask - bid
T = dur.sum()
print('time one-sided or empty share', dur[~two].sum() / T, 'bid empty', dur[np.isnan(bid)].sum() / T, 'ask empty', dur[np.isnan(ask)].sum() / T)
# episodes of one side empty
for name, m in [('bid empty', np.isnan(bid)), ('ask empty', np.isnan(ask))]:
    st = np.flatnonzero(m & ~np.r_[False, m[:-1]]); en = np.flatnonzero(m & ~np.r_[m[1:], False])
    d = (t[np.minimum(en + 1, len(t) - 1)] - t[st]) / 1e3
    print(name, 'episodes', len(st), 'per s', len(st) / T, 'dur ms quantiles', np.percentile(d, [10, 50, 90, 99]).round(2), 'max', d.max())
w = dur * two
tab = pd.Series(w).groupby(np.where(two, sp, -1)).sum() / w.sum()
print('spread time share', tab.loc[tab.index > 0].head(8).round(4).to_dict())
bq, aq = q.bid_qty.to_numpy(), q.ask_qty.to_numpy()
print('touch qty mean (time-weighted, both sides)', (w * (bq + aq) / 2).sum() / w.sum())
seg = (t // 600e6).astype(int)
rows = []
for s in np.unique(seg):
    m = seg == s
    ww = w[m]
    rows.append({'seg': s, 'sp1': (ww * (sp[m] == 1)).sum() / ww.sum(), 'sp2': (ww * (sp[m] == 2)).sum() / ww.sum(),
                 'sp3+': (ww * (sp[m] >= 3)).sum() / ww.sum(), 'onesided': (dur[m] * ~two[m]).sum() / dur[m].sum(),
                 'touchq': (ww * (bq[m] + aq[m]) / 2).sum() / ww.sum()})
print(pd.DataFrame(rows).round(4).to_string())
