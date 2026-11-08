"""Cross-check makers' 15-lot-agent drift claim; calm momentum inside vs outside its windows (exact S residual)."""
import numpy as np, pandas as pd
from common import *
d=pd.read_pickle('../makers/cache/r15_v2.pkl'); d=d[d.L>=2]
w=d.groupby('chain').agg(t0=('t_us','min'),t1=('end_t_us','max'),sg=('sg','first')); w=w/[1e6,1e6,1]; w['sg']=w.sg.astype(int)
t00=pd.read_parquet('cache/round4_orders.parquet').t.iloc[0]
B=0.5; r=np.load('cache/round4_sresid05.npy'); nb=len(r); v=regime_1s('round4'); vb=v[(np.arange(nb)*B).astype(int)]
x=L('round4','strue').sp.to_numpy(); S=lambda t: np.interp(t,np.arange(len(x))*0.1,x)
# NOTE t_us in makers table: check offset vs my t (seconds from first event)
print('first window',w.iloc[0].to_dict())
inw=np.zeros(nb,bool)
for _,e in w.iterrows(): inw[int(e.t0/B):int(e.t1/B)+1]=True
print('windows',len(w),'calm time share in windows',(inw&~vb).sum()/(~vb).sum())
ok=~np.isnan(r)
for nm,m in [('calm in windows',ok&~vb&inw),('calm outside',ok&~vb&~inw),('vol',ok&vb)]:
    xx=np.where(m,r,0.0); ac=[]
    for l in [1,2,4,6,8]:
        mm=m[l:]&m[:-l]; ac.append(np.sum(xx[l:][mm]*xx[:-l][mm])/np.sqrt(np.sum(xx[l:][mm]**2)*np.sum(xx[:-l][mm]**2)))
    print(f'{nm:16s} bins={m.sum()} resid sd={xx[m].std():.3f} ACF 0.5/1/2/3/4 s={np.round(ac,3).tolist()}')
sg=w.sg.to_numpy(); t0=w.t0.to_numpy(); t1=w.t1.to_numpy()
print('signed S move during windows mean',np.mean(sg*(S(t1)-S(t0))).round(2),' per s',np.mean(sg*(S(t1)-S(t0))/(t1-t0)).round(2), ' dur median',np.median(t1-t0).round(1))
print('signed pre (−3..0 s)',np.mean(sg*(S(t0)-S(t0-3))).round(2),' post (end..+5)',np.mean(sg*(S(t1+5)-S(t1))).round(2))
