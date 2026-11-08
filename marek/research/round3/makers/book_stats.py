"""Book by state on a 10 ms grid: spread distribution, touch qty, mid - S, and depth by level (rel. to the same-side touch)
split into ladder (2 ms chains) / other makers. Writes book_by_state.csv and depth_by_level.csv."""
import numpy as np, pandas as pd
from s_load import with_s
import states
o = with_s()
q = pd.read_pickle('cache/quotes.pkl')
grid = np.arange(0, 6000_000_000, 10_000)
j = np.searchsorted(q.t_us.to_numpy(), grid, 'right') - 1
ok = j >= 0
bid = q.bid.to_numpy().astype(float)[j]; ask = q.ask.to_numpy().astype(float)[j]
bq = q.bid_qty.to_numpy().astype(float)[j]; aq = q.ask_qty.to_numpy().astype(float)[j]
bid[(bid == 0)] = np.nan; ask[ask == 0] = np.nan
k = np.clip(np.searchsorted(o.t_us.to_numpy(), grid, 'right') - 1, 0, len(o) - 1)
S = o.S.to_numpy()[k]
st = states.state(grid)
sp = ask - bid; mid = (ask + bid) / 2
rows = []
for s_ in ['calm', 'vol', 'drought', 'post', 'all']:
    m = (st == s_) if s_ != 'all' else np.ones(len(grid), bool)
    m &= ~np.isnan(sp)
    x = sp[m]
    rows.append(dict(state=s_, time_s=m.sum() / 100, sd_mid_S=np.nanstd(mid[m] - S[m]), sp1=(x == 1).mean(), sp2=(x == 2).mean(), sp3p=(x >= 3).mean(),
                     mean_spread=x.mean(), touch_qty=np.nanmean((bq[m] + aq[m]) / 2), S_minus_bid=np.nanmean(S[m] - bid[m]), ask_minus_S=np.nanmean(ask[m] - S[m]),
                     one_sided_share=np.isnan(sp[(st == s_) if s_ != 'all' else slice(None)]).mean()))
df = pd.DataFrame(rows).round(4); pd.set_option('display.width', 220); print(df.to_string()); df.to_csv('book_by_state.csv', index=False)
# depth by level: sample every 100 ms
G = grid[::10]; stG = st[::10]; bidG = bid[::10]; askG = ask[::10]
m = o[~o.ioc].copy(); m['end_t'] = m.end_t_us.fillna(6e9).astype(np.int64)
t0 = m.t_us.to_numpy(); t1 = m.end_t.to_numpy()
i0 = np.searchsorted(G, t0); i1 = np.searchsorted(G, t1)       # grid points in [t0, t1)
n = np.maximum(i1 - i0, 0)
rep = np.repeat(np.arange(len(m)), n); gi = np.repeat(i0, n) + (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n))
sg = m.sg.to_numpy()[rep]; pr = m.price.to_numpy()[rep]
rem = (m.volume.to_numpy() - 0)[rep]
rel = np.where(sg > 0, pr - bidG[gi], askG[gi] - pr)
lad = m.lad.to_numpy()[rep]
D = pd.DataFrame(dict(st=stG[gi], rel=-rel, lad=lad, q=rem))
D = D[(D.rel >= 0) & (D.rel <= 12)]
nG = pd.Series(stG).value_counts()
tab = D.groupby(['st', 'rel', 'lad']).q.sum().unstack(fill_value=0)
tab = tab.div(nG.reindex(tab.index.get_level_values(0)).to_numpy() * 2, axis=0).round(1)   # per side per snapshot
tab.columns = ['other', 'ladder_chain']; tab['ladder_share'] = (tab.ladder_chain / (tab.other + tab.ladder_chain)).round(3)
print(tab.to_string()); tab.to_csv('depth_by_level.csv')
