import numpy as np, pandas as pd
from common import *
for n,src in [('round3','strue'),('round4','strue')]:
    o=L(n,'orders'); sh=shocks(n); x=L(n,src).sp.to_numpy()
    def at(t): return x[np.clip((np.asarray(t)*10).astype(int),0,len(x)-1)]
    on=sh.onset.to_numpy(); J=at(on+4)-at(on+2.5)
    rows=[]
    for a in on:
        w=o[(o.t>=a)&(o.t<a+2.9)]; k=w[w.ioc]; m=w[~w.ioc]
        kb=(k.side==0).sum()-(k.side==1).sum(); kv=(np.where(k.side==0,1,-1)*k.vol).sum()
        mb=(m.side==0).sum()-(m.side==1).sum()
        pre=w[w.t<a]
        rows.append((kb,kv,mb,at(a+2.9)-at(a-5),at(a)-at(a-30)))
    R=np.array(rows,float)
    names=['drought taker count imb','drought taker vol imb','drought maker side imb','S move onset-5..+2.9','S move -30..0']
    print(n,'corr with J (n=27):',{nm:round(np.corrcoef(R[:,i],J)[0,1],3) for i,nm in enumerate(names)})
    # taker size == touch qty
    k=o[o.ioc]; tq=np.where(k.side==0,k.paq,k.pbq)
    print('   P(taker vol == opposite touch qty)=',(k.vol==tq).mean().round(5),' P(vol==touch qty of random other taker)=',(k.vol.to_numpy()==np.roll(tq,7)).mean().round(5))
