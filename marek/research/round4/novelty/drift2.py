import numpy as np, pandas as pd
from common import *
for n in ['round3','round4']:
    r=np.load(f'cache/{n}_resid05.npy'); v=regime_1s(n)
    d=np.diff(np.r_[0,v[:6000].astype(int),0]); segs=list(zip(np.where(d==1)[0],np.where(d==-1)[0]))
    rows=[]
    for a,b in segs:
        if b-a<15: continue
        x=r[2*a:2*b]
        if np.isnan(x).mean()>0.1: continue
        c=np.nancumsum(x); tt=np.arange(len(c))*0.5
        sl=np.polyfit(tt,c,1)[0]; R2=1-np.var(c-np.polyval(np.polyfit(tt,c,1),tt))/np.var(c)
        # half split slopes
        h=len(c)//2; s1=np.polyfit(tt[:h],c[:h],1)[0]; s2=np.polyfit(tt[h:],c[h:],1)[0]
        rows.append((a,b-a,c[-1],sl,R2,s1,s2))
    R=pd.DataFrame(rows,columns=['start','dur','tot','slope','R2','s1','s2'])
    print(n,'vol segs',len(R),' mean R2 of linear trend',R.R2.mean().round(3),' corr(first-half slope, second-half slope)',np.corrcoef(R.s1,R.s2)[0,1].round(3),' |tot| mean',R.tot.abs().mean().round(1))
    if n=='round4':
        print(R.round(2).to_string())
        # print path for 3 longest segs every 2s
        for _,s in R.sort_values('dur').tail(3).iterrows():
            x=np.nancumsum(r[2*int(s.start):2*int(s.start+s.dur)]); print(int(s.start),np.round(x[::4],0).astype(int).tolist())
