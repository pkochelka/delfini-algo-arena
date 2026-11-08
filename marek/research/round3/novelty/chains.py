"""Build 2000us chains of maker inserts in round3 and save per-order chain ids."""
import numpy as np, pandas as pd
o=pd.read_parquet('cache/round3_orders.parquet')
o['tus']=np.rint(o.t.to_numpy()*1e6).astype(np.int64)
m=o[~o.ioc].copy()
pos={t:i for i,t in enumerate(m.tus.to_numpy())}  # assumes unique timestamps among makers
assert len(pos)==len(m)
tus=m.tus.to_numpy(); side=m.side.to_numpy()
prev=np.full(len(m),-1)
for i,t in enumerate(tus):
    j=pos.get(t-2000)
    if j is not None: prev[i]=j
print('linked',(prev>=0).sum(),'same side',(side[prev[prev>=0]]==side[prev>=0]).mean())
# chain id
cid=np.full(len(m),-1); idx=np.zeros(len(m),int); c=0
for i in range(len(m)):
    if prev[i]>=0: cid[i]=cid[prev[i]]; idx[i]=idx[prev[i]]+1
    else: cid[i]=c; c+=1
m['cid']=cid; m['cidx']=idx
sz=m.groupby('cid').size(); multi=sz[sz>=2].index
m['chain']=m.cid.isin(multi)
m['clen']=m.cid.map(sz)
m.to_parquet('cache/round3_makers_chains.parquet')
print('chains len>=2:',len(multi),'orders in chains',m.chain.sum())
print('chain length dist', sz[sz>=2].value_counts().sort_index().to_dict())
