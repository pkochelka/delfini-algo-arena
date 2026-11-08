"""Build data/.cache/round3_S_oracle.npz: exact S after each insert (a taker row: after its own kick), regime, shock. Run from the repo root:
uv run --with pyarrow python marek/research/round3/ideas/ceiling/ceiling_mk_oracle.py"""
import numpy as np, pandas as pd
d = pd.read_parquet('marek/research/round3/s_dynamics/s_path.parquet')
tp = pd.read_csv('data/round3.csv')
ins = tp[tp.Operation == 'Insert'].reset_index(drop=True)
assert len(ins) == len(d)
assert (np.round(ins.Time.values * 1e6).astype(np.int64) == d.t_us.values).mean() > 0.999
tk = (ins.Lifespan.values == 'F')
s = d.s_hat.values
sa = s.copy()
sa[:-1][tk[:-1]] = s[1:][tk[:-1]]
np.savez('data/.cache/round3_S_oracle.npz', t_us=d.t_us.values.astype(np.int64), S=sa, vol=d.vol.values.astype(np.int8), shock=d.shock.values.astype(np.int8), s_sd=d.s_sd.values)
print(len(sa), tk.mean())
