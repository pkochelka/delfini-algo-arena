"""Periodic takers: IOC takers linked by an exact multiple of 500000 us to an earlier IOC taker.
Writes cache/tw.pkl (takers with chain id, slot, side...)."""
import numpy as np, pandas as pd
from prep import load
o = load().reset_index(drop=True)
tk = o[o.ioc].copy()
t = tk.t_us.to_numpy()
idx = tk.index.to_numpy()
pos = pd.Series(np.arange(len(t)), index=t)
for d in [500000, 1000000, 1500000, 2000000, 2500000, 3000000]:
    n = pos.reindex(t - d).notna().sum()
    print('gap', d, 'links', n)
# phase test: t mod 500000 distribution for linked takers
link = pos.reindex(t - 500000).to_numpy()
lk = ~np.isnan(link)
ph = t % 500000
print('phase of linked takers (mod 0.5 s) unique count', len(np.unique(ph[lk])), 'top', pd.Series(ph[lk]).value_counts().head(5).to_dict())
# chains with gap 500000 or 1000000 (allow one skipped slot)
pred = np.full(len(t), -1)
for d in [500000, 1000000, 1500000]:
    p = pos.reindex(t - d).to_numpy()
    m = (pred < 0) & ~np.isnan(p)
    pred[m] = p[m].astype(int)
chain = np.arange(len(t)); k = np.zeros(len(t), int)
for i in range(len(t)):
    if pred[i] >= 0:
        chain[i] = chain[pred[i]]; k[i] = k[pred[i]] + 1
tk['tw_pred'] = pred; tk['tw_chain'] = chain; tk['tw_k'] = k
tk['tw_L'] = tk.groupby('tw_chain').tw_k.transform('max') + 1
c = tk[tk.tw_L >= 2]
print('takers in periodic chains', len(c), 'chains', c.tw_chain.nunique())
g = c.groupby('tw_chain').agg(t0=('t_us', 'min'), t1=('t_us', 'max'), n=('t_us', 'size'), sg=('sg', 'mean'),
                              vol=('volume', 'median'), volsum=('volume', 'sum'), filled=('filled', 'sum'),
                              vmin=('volume', 'min'), vmax=('volume', 'max'))
g['t0'] /= 1e6; g['t1'] /= 1e6; g['dur'] = g.t1 - g.t0
g['phase_us'] = (g.t0 * 1e6).round().astype(np.int64) % 500000
pd.set_option('display.width', 250)
print(g.sort_values('t0').to_string())
tk.to_pickle('cache/tw.pkl')
