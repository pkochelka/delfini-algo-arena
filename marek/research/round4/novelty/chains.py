"""Build 2000us chains of maker inserts and save per-order chain ids (round4)."""
import sys, numpy as np, pandas as pd
n=sys.argv[1] if len(sys.argv)>1 else 'round4'
o=pd.read_parquet(f'cache/{n}_orders.parquet')
o['tus']=np.rint(o.t.to_numpy()*1e6).astype(np.int64)
m=o[~o.ioc].copy()
tus=m.tus.to_numpy(); side=m.side.to_numpy()
u,cn=np.unique(tus,return_counts=True); print('dup maker ts',(cn>1).sum())
pos={t:i for i,t in enumerate(tus)}
prev=np.full(len(m),-1)
for i,t in enumerate(tus):
    j=pos.get(t-2000)
    if j is not None: prev[i]=j
print('linked',(prev>=0).sum(),'same side',(side[prev[prev>=0]]==side[prev>=0]).mean())
cid=np.full(len(m),-1); idx=np.zeros(len(m),int); c=0
for i in range(len(m)):
    if prev[i]>=0: cid[i]=cid[prev[i]]; idx[i]=idx[prev[i]]+1
    else: cid[i]=c; c+=1
m['cid']=cid; m['cidx']=idx
sz=m.groupby('cid').size(); multi=sz[sz>=2].index
m['chain']=m.cid.isin(multi); m['clen']=m.cid.map(sz)
m.to_parquet(f'cache/{n}_makers_chains.parquet')
print('chains len>=2:',len(multi),'orders in chains',m.chain.sum())
print('chain length dist', sz[sz>=2].value_counts().sort_index().to_dict())
