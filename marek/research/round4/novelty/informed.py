import numpy as np, pandas as pd
from common import *
H=[0.001,0.05,0.5,2,5,15,30,60]
for n in ['round3','round4']:
    o=L(n,'orders'); k=o[o.ioc].copy(); v=regime_1s(n); sh=shocks(n)
    top=np.load(f'cache/{n}_top.npz'); tt=top['t']; b=top['bid'].astype(float); a=top['ask'].astype(float); b[b<=0]=np.nan;a[a<=0]=np.nan; mid=pd.Series((a+b)/2).ffill().to_numpy()
    def midat(t): return mid[np.clip(np.searchsorted(tt,t,side='right')-1,0,len(mid)-1)]
    bad=np.zeros(len(k),bool)
    for x in sh.onset: bad|=(k.t>x-62)&(k.t<x+8)
    k=k[~bad & (k.t<5930)]
    sg=np.where(k.side==0,1,-1); m0=mid[k.row.to_numpy()-1]
    filled=k.filled>0
    print(n, 'n',len(k))
    for r in [False,True]:
        for f in [True,False]:
            sel=(v[k.t.astype(int)]==r)&(filled==f)
            out=[np.nanmean(sg[sel]*(midat(k.t.to_numpy()[sel]+h)-m0[sel])) for h in H]
            print(f'  vol={r} filled={f} n={sel.sum()}', ' '.join(f'{h}s:{x:+.3f}' for h,x in zip(H,out)))
