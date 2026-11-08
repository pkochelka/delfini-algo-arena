"""Per-insert class labels for round 4 (row order = tape inserts = s_path.parquet rows).
cls: 0 base maker, 1 ladder (exact-2000 us chain, L >= 2), 2 r15 agent, 3 regular IOC taker, 4 TWAP child.
Writes cache/order_class.parquet (t_us, cls, r15_chain, tw_chain, lad_chain)."""
import numpy as np, pandas as pd
o = pd.read_pickle('cache/orders.pkl')
ch = pd.read_pickle('cache/chains.pkl'); a = pd.read_pickle('cache/r15_v2.pkl'); tw = pd.read_pickle('cache/tw.pkl')
cls = np.zeros(len(o), np.int8)
cls[(ch.L >= 2).to_numpy() & ~o.ioc.to_numpy()] = 1
cls[a.index] = 2
cls[o.ioc.to_numpy()] = 3
cls[tw.index[tw.tw_L >= 2]] = 4
out = pd.DataFrame({'t_us': o.t_us.to_numpy(), 'cls': cls})
out['r15_chain'] = -1; out.loc[a.index, 'r15_chain'] = a.chain.to_numpy()
out['tw_chain'] = -1; tt = tw[tw.tw_L >= 2]; out.loc[tt.index, 'tw_chain'] = tt.tw_chain.to_numpy()
out['lad_chain'] = np.where(cls == 1, ch.chain.to_numpy(), -1)
out.to_parquet('cache/order_class.parquet')
print(out.cls.value_counts().sort_index().to_dict())
