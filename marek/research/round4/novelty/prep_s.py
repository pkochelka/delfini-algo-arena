"""S on 0.1 s grid (last insert's s_hat at or before t) and S at each insert, from s_dynamics/s_path.parquet."""
import sys, numpy as np, pandas as pd
n=sys.argv[1] if len(sys.argv)>1 else 'round4'
p=pd.read_parquet(f'/Users/marek/Documents/delfini-algo-arena/marek/research/{n}/s_dynamics/s_path.parquet')
o=pd.read_parquet(f'cache/{n}_orders.parquet'); assert len(p)==len(o)
t0us=None
t=o.t.to_numpy(); s=p.s_hat.to_numpy()
pd.DataFrame(dict(s=s,s_filt=p.s_filt.to_numpy(),vol=p.vol.to_numpy())).to_parquet(f'cache/{n}_s_at_insert.parquet')
grid=np.arange(0,6000,0.1); i=np.searchsorted(t,grid,side='right')-1
pd.DataFrame(dict(t=grid,sp=s[np.clip(i,0,len(s)-1)])).to_parquet(f'cache/{n}_strue.parquet')
print(p.columns.tolist(), len(p), 'vol frac', p.vol.mean())
