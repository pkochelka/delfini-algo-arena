import numpy as np, pandas as pd
from common import *
c={}
for n in ['round3','round4']:
    o=L(n,'orders'); m=o[~o.ioc]; ch=pd.read_parquet(f'cache/{n}_makers_chains.parquet')[['oid','chain']]
    m=m.merge(ch,on='oid',how='left'); m=m[~m.chain]
    c[n]=m.vol.value_counts().reindex(np.arange(1,3001)).fillna(0)
    print(n,len(m))
r=c['round4']; s=c['round3']*(r.sum()/c['round3'].sum())
z=(r-s)/np.sqrt(r+s+1)
print('top excess sizes r4 vs r3 (size r4 r3scaled z):'); print(pd.DataFrame(dict(r4=r,r3=s.round(0),z=z.round(1))).sort_values('z',ascending=False).head(15).to_string())
print('top deficit:'); print(pd.DataFrame(dict(r4=r,r3=s.round(0),z=z.round(1))).sort_values('z').head(8).to_string())
# same for takers
for n in ['round3','round4']:
    o=L(n,'orders'); c[n]=o[o.ioc].vol.value_counts().reindex(np.arange(1,3001)).fillna(0)
r=c['round4']; s=c['round3']*(r.sum()/c['round3'].sum()); z=(r-s)/np.sqrt(r+s+1)
print('TAKERS top excess:'); print(pd.DataFrame(dict(r4=r,r3=s.round(0),z=z.round(1))).sort_values('z',ascending=False).head(8).to_string())
