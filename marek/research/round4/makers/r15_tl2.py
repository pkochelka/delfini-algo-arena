import sys, numpy as np, pandas as pd
o = pd.read_pickle('cache/orders.pkl')
a = pd.read_pickle('cache/r15_v2.pkl')
S = pd.read_parquet('../s_dynamics/s_path.parquet')
t0, t1 = float(sys.argv[1]) * 1e6, float(sys.argv[2]) * 1e6
w = o[(o.t_us >= t0) & (o.t_us <= t1)].copy()
w['s'] = S.s_hat.reindex(w.index).round(2)
w['chain'] = a.chain.reindex(w.index); w['k'] = a.k.reindex(w.index)
w = w[w.ioc | w.chain.notna() | (w.volume == 15)]
w['life_ms'] = ((w.end_t_us - w.t_us) / 1e3).round(1); w['t'] = w.t_us / 1e6
pd.set_option('display.width', 250)
print(w[['t', 'sg', 'price', 'volume', 'ioc', 'filled', 'end', 'life_ms', 'bid', 'ask', 'bid_qty', 'ask_qty', 's', 'chain', 'k']].to_string())
