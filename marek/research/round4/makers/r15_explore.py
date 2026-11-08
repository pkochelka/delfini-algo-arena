"""Explore maker inserts exactly 1 us after an IOC taker (the new 15-lot reactive maker)."""
import numpy as np, pandas as pd
from prep import load
o = load().reset_index(drop=True)
t = o.t_us.to_numpy(); ioc = o.ioc.to_numpy()
nxt = np.r_[t[1:] - t[:-1], 10**9]
# event rows: we need any row (incl cancels) between; use orders table (inserts only) — check rows adjacency too
row = o.row.to_numpy()
i = np.where(ioc[:-1] & (~ioc[1:]) & (nxt[:-1] == 1))[0]
j = i + 1
print('pairs taker->maker 1us', len(i), 'rows adjacent', (row[j] - row[i] == 1).mean())
T = o.iloc[i].reset_index(drop=True); M = o.iloc[j].reset_index(drop=True)
print('maker vol counts', M.volume.value_counts().head(10).to_dict())
print('side opposite', (M.sg != T.sg).mean())
# price vs taker limit
print('M.price - T.price', (M.price - T.price).value_counts().sort_index().to_dict())
print('taker filled frac', (T.filled > 0).mean(), 'taker filled==vol', (T.filled == T.volume).mean())
print('taker vol quantiles', T.volume.quantile([.1,.25,.5,.75,.9]).to_dict())
# taker limit vs touch before
tl = np.where(T.sg > 0, T.price - T.ask, T.bid - T.price)
print('taker limit beyond opposite touch', pd.Series(tl).value_counts().sort_index().to_dict())
# maker price vs touch before taker (same side as maker)
mp = np.where(M.sg > 0, M.price - T.bid, T.ask - M.price)   # >0 = inside/improving
print('maker price vs pre-taker same-side touch (+ better)', pd.Series(mp).value_counts().sort_index().to_dict())
mp2 = np.where(M.sg > 0, M.price - M.bid, M.ask - M.price)
print('maker price vs post-taker same-side touch (+ better)', pd.Series(mp2).value_counts().sort_index().to_dict())
mp3 = np.where(M.sg > 0, M.ask - M.price, M.price - M.bid)
print('maker price distance to post-taker opposite touch', pd.Series(mp3).value_counts().sort_index().to_dict())
print('maker end', M.end.value_counts().to_dict(), 'filled>0', (M.filled > 0).mean())
life = (M.end_t_us - M.t_us) / 1e6
print('maker life quantiles', life.quantile([.1,.5,.9]).to_dict())
print('time quantiles', (T.t_us/1e6).describe())
pd.DataFrame({'t': T.t_us/1e6, 'tsg': T.sg, 'tvol': T.volume, 'tfill': T.filled, 'tpx': T.price, 'bid': T.bid, 'ask': T.ask,
              'mvol': M.volume, 'msg': M.sg, 'mpx': M.price, 'mbid': M.bid, 'mask': M.ask, 'mfill': M.filled, 'mend': M.end,
              'mlife': life}).to_csv('cache/pairs1us.csv', index=False)
