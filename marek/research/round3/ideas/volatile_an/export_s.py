"""Export exact S (s_hat) and the causal s_filt from s_dynamics/s_path.parquet to spath.npz / sfilt.npz in the cwd.
uv run --with pyarrow --with pandas python export_s.py"""
import pyarrow.parquet as pq, numpy as np
P = '/Users/marek/Documents/delfini-algo-arena/marek/research/round3/s_dynamics/s_path.parquet'
t = pq.read_table(P, columns=['t_us', 's_hat', 's_filt', 'vol', 'shock']).to_pandas()
np.savez('spath.npz', t=t.t_us.values * 1e-6, s=t.s_hat.values, vol=t.vol.values, shock=t.shock.values)
np.savez('sfilt.npz', t=t.t_us.values * 1e-6, f=t.s_filt.values)
