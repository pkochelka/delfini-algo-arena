import numpy as np, pandas as pd
from common import *
o=L('round4','orders'); k=o[o.ioc].copy()
k['tus']=np.rint(k.t*1e6).astype(np.int64)
a=k[k.vol==160].copy()
d=np.diff(a.tus.to_numpy()); print('exact dt counts top', pd.Series(d[d<2_000_000]).value_counts().head(10).to_dict())
# episodes: gap > 2s
ep=np.cumsum(np.r_[True,np.diff(a.t.to_numpy())>2.0]); a['ep']=ep
g=a.groupby('ep').agg(t0=('t','min'),t1=('t','max'),n=('t','size'),side=('side','mean'),fill=('filled','sum'))
print('episodes',len(g)); print(g.n.value_counts().sort_index().to_dict())
print(g.head(60).to_string())
