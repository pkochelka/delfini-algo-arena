"""Attach S (s_dynamics s_path.parquet, value just before the insert's own impact) to takers -> cache/takers_s.parquet.
Also S just before the next/prev taker and S at grid offsets via the insert stream."""
import numpy as np, pandas as pd
from pathlib import Path
D = Path(__file__).parent; C = D / 'cache'
sp = pd.read_parquet(D.parent / 's_dynamics/s_path.parquet')
r = pd.read_pickle(C / 'rows.pkl'); ins = r[r.op == 0].reset_index(drop=True)
assert len(sp) == len(ins)
tk = pd.read_parquet(C / 'takers.parquet')
m = ins.ioc.to_numpy()
for c in ['s_hat', 's_sd', 's_filt', 'shock']: tk[c] = sp[c].to_numpy()[m]
tk['svol'] = sp['vol'].to_numpy()[m]
tk.to_parquet(C / 'takers_s.parquet')
# compact S series on all inserts for lookups
np.savez(C / 's_series.npz', t=ins.t.to_numpy(), s=sp.s_hat.to_numpy(), sf=sp.s_filt.to_numpy(), sd=sp.s_sd.to_numpy())
print(tk[['s_hat', 's_sd', 'px', 'mid']].describe().T)
print('sdyn vol vs reg', (tk.svol == tk.reg).mean())
