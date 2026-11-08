"""Per-taker table -> cache/takers.parquet: t, sg (+1 buy), vol, px, filled, bid/ask before, mid, regime(1s), shock flags."""
import numpy as np, pandas as pd
from pathlib import Path
D = Path(__file__).parent; C = D / 'cache'
r = pd.read_pickle(C / 'rows.pkl'); o = pd.read_pickle(C / 'orders.pkl')
ins = r[r.op == 0].reset_index(drop=True)
assert len(ins) == len(o)
tk = ins[ins.ioc.to_numpy()].copy(); ot = o[o.ioc.to_numpy()]
tk['filled'] = ot.filled.to_numpy(); tk['oid'] = tk.oid.astype(np.int64); tk['sg'] = np.where(tk.side == 0, 1, -1)
tk['mid'] = (tk.bid_before + tk.ask_before) / 2
vol = np.load(C / 'vol1s.npy'); tk['vol1s'] = vol[np.minimum(tk.t.astype(int), 5999)]
ev = pd.read_csv(D / 'shock_events.csv')
ph = np.full(len(tk), 1e9)
for on in ev.onset:
    d = tk.t.to_numpy() - on; m = (d > -5) & (d < 30); ph[m] = d[m]
tk['shock_dt'] = ph
tw = pd.read_parquet(D.parent / 'novelty/cache/round4_takers_tw.parquet')
assert (tw.oid.to_numpy() == tk.oid.to_numpy()).all()
tk['tw'] = tw.tw.to_numpy(); tk['tw_cid'] = tw.tw_cid.to_numpy()
vs = pd.read_csv(D.parent / 's_dynamics/vol_segments.csv'); v = np.zeros(len(tk), int)
for a, b in zip(vs.start, vs.end): v[(tk.t >= a) & (tk.t < b)] = 1
tk['reg'] = v
tk = tk[['t', 'sg', 'vol', 'px', 'filled', 'bid_before', 'ask_before', 'mid', 'vol1s', 'reg', 'shock_dt', 'tw', 'tw_cid']].reset_index(drop=True)
print('agree vol1s vs s_dynamics seg', (tk.vol1s == tk.reg).mean())
tk.to_parquet(C / 'takers.parquet')
print(tk.describe().T)
