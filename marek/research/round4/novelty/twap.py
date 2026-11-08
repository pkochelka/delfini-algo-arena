"""Build TWAP episodes: chains of 160-lot IOC takers exactly 500000 us apart. Save table + per-taker flag."""
import numpy as np, pandas as pd
from common import *
o=L('round4','orders'); k=o[o.ioc].copy(); k['tus']=np.rint(k.t*1e6).astype(np.int64)
pos={t:i for i,t in enumerate(k.tus.to_numpy())}
tus=k.tus.to_numpy(); vol=k.vol.to_numpy(); side=k.side.to_numpy()
prev=np.full(len(k),-1)
for i in np.where(vol==160)[0]:
    j=pos.get(tus[i]-500000)
    if j is not None and vol[j]==160: prev[i]=j
cid=np.full(len(k),-1); idx=np.zeros(len(k),int); c=0
for i in np.where(vol==160)[0]:
    if prev[i]>=0: cid[i]=cid[prev[i]]; idx[i]=idx[prev[i]]+1
    else: cid[i]=c; c+=1
k['tw_cid']=cid; k['tw_idx']=idx
sz=k[cid>=0].groupby('tw_cid').size()
k['tw']=k.tw_cid.map(sz).fillna(0)>=3
print('chains len>=2',(sz>=2).sum(),'len>=3',(sz>=3).sum(),'orders in len>=3',k.tw.sum(), 'len dist',sz[sz>=2].value_counts().sort_index().to_dict())
tw=k[k.tw]
g=tw.groupby('tw_cid').agg(t0=('t','min'),t1=('t','max'),n=('t','size'),sidem=('side','mean'),filled=('filled','sum'),vol=('vol','sum'),ph0=('price','first'),ph1=('price','last'))
g['side']=np.where(g.sidem<0.5,'B','A'); g['dur']=g.t1-g.t0
print(g.to_string())
print('mixed side chains',((g.sidem>0)&(g.sidem<1)).sum(),' dur stats',g.dur.describe().round(2).to_dict())
print('t0 mod 0.5 / start us pattern', (np.rint(g.t0*1e6)%500000).describe())
g.to_csv('twap_episodes.csv')
k[['row','oid','t','side','vol','price','filled','tw','tw_cid','tw_idx']].to_parquet('cache/round4_takers_tw.parquet')
# gaps between episodes
st=g.t0.to_numpy(); en=g.t1.to_numpy(); print('gaps end->next start',np.round(st[1:]-en[:-1],1).tolist())
print('total TWAP time',g.dur.sum()+0.5*len(g),'frac',(g.dur.sum()+0.5*len(g))/6000)
