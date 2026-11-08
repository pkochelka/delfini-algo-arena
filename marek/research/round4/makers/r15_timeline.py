import sys, numpy as np, pandas as pd
from prep import load
o = load().reset_index(drop=True)
q = pd.read_pickle('cache/r15_chains.pkl')
t0, t1 = float(sys.argv[1]) * 1e6, float(sys.argv[2]) * 1e6
w = o[(o.t_us >= t0) & (o.t_us <= t1) & (o.ioc | ((o.volume == 15)))].copy()
w['chain'] = q.chain.reindex(w.index); w['k'] = q.k.reindex(w.index); w['L'] = q.L.reindex(w.index)
w['life_ms'] = (w.end_t_us - w.t_us) / 1e3
w['t'] = w.t_us / 1e6
pd.set_option('display.width', 250)
print(w[['t', 'sg', 'price', 'volume', 'ioc', 'filled', 'end', 'life_ms', 'bid', 'ask', 'bid_qty', 'ask_qty', 'chain', 'k', 'L']].to_string())
