"""Predictability of future mid (and S proxy) from past moves / taker imbalance / TWAP state, r3 vs r4, by regime."""
import numpy as np, pandas as pd
from common import *
P=[0.5,1,2,5,10,20]; H=[1,2,5,10,20,30,60]
for n in ['round3','round4']:
    g=L(n,'grid'); mid=g.mid.ffill().bfill().to_numpy(); N=len(mid); t=g.t.to_numpy()
    v=regime_1s(n); o=L(n,'orders'); k=o[o.ioc]
    bad=np.zeros(N,bool)
    for x in shocks(n).onset: bad|=(t>x-1)&(t<x+30)
    bad|=t<25
    sg=np.where(k.side==0,1,-1); bi=(k.t*10).astype(int).clip(0,N-1)
    q=np.cumsum(np.bincount(bi,weights=sg,minlength=N))
    tw=np.zeros(N)
    if n=='round4':
        e=pd.read_csv('twap_episodes.csv')
        for _,r in e.iterrows(): tw[int(r.t0*10)+11:int(r.t1*10)+1]=1 if r.side=='B' else -1   # known after 2 children
    a=np.arange(250,N-610,5); a=a[~bad[a]&~bad[np.minimum(a+600,N-1)]]
    reg=v[(t[a]).astype(int)]
    print('=====',n)
    for R in [False,True]:
        aa=a[reg==R]
        rows=[]
        feats={f'dmid{p}':mid[aa]-mid[aa-int(p*10)] for p in P}
        feats.update({f'imb{p}':q[aa]-q[aa-int(p*10)] for p in [1,5,20]})
        if n=='round4': feats['twap']=tw[aa]
        for h in H:
            fut=mid[aa+h*10]-mid[aa]
            rows.append({f:np.corrcoef(x,fut)[0,1] if x.std()>0 else np.nan for f,x in feats.items()}|{'H':h})
        df=pd.DataFrame(rows).set_index('H'); print(' vol' if R else ' calm', 'n=',len(aa)); print(df.round(3).to_string())
