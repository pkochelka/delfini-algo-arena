import numpy as np, pandas as pd
m=pd.read_parquet('cache/round3_makers_chains.parquet')
m['touchoff']=np.where(m.side==0, m.pbid-m.price, m.price-m.pask)
m['life']=m.end_t-m.t
C=m[m.chain].sort_values(['cid','cidx'])
g=C.groupby('cid').agg(start=('touchoff','first'),end=('touchoff','last'),n=('vol','size'),t=('t','first'),side=('side','first'),
   vmean=('vol','mean'),life=('life','mean'), sumv=('vol','sum'))
print(pd.crosstab(g.start.clip(0,8),g.n.clip(1,8)))
print(pd.crosstab(g.end.clip(0,9),g.n.clip(1,8)))
g.to_parquet('cache/round3_chains.parquet')
# sizes: within chain sizes correlated?
C['lv']=np.log(C.vol)
cm=C.groupby('cid').lv.transform('mean')
print('within-chain lv sd',(C.lv-cm).std(),' between chains sd of mean',C.groupby('cid').lv.mean().std(), 'overall',C.lv.std())
print('chain order lv by cidx',C.groupby('cidx').lv.mean().round(3).to_dict())
print('vol by touchoff', C.groupby(C.touchoff.clip(0,8)).vol.median().to_dict())
