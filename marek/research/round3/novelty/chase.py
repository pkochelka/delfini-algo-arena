import numpy as np, pandas as pd
from common import *
H=[0.05,0.2,0.5,1,2,5,15]
for n in ['round2','round3']:
    o=L(n,'orders'); k=o[o.ioc].copy(); v=regime_1s(n); sh=shocks(n)
    top=np.load(f'cache/{n}_top.npz'); tt=top['t']; b=top['bid'].astype(float); a=top['ask'].astype(float); b[b<=0]=np.nan;a[a<=0]=np.nan; mid=pd.Series((a+b)/2).ffill().to_numpy()
    def midat(t): return mid[np.clip(np.searchsorted(tt,t,side='right')-1,0,len(mid)-1)]
    bad=np.zeros(len(k),bool)
    for x in sh.onset: bad|=(k.t>x-2)&(k.t<x+25)
    k=k[~bad & (k.t>20)]
    sg=np.where(k.side==0,1,-1); m0=mid[k.row.to_numpy()-1]
    print(n)
    for r in [False,True]:
        sel=(v[k.t.astype(int)]==r)
        out=[]
        for h in H:
            dm=m0[sel]-midat(k.t.to_numpy()[sel]-h)
            out.append((np.corrcoef(sg[sel],dm)[0,1], np.mean(sg[sel]*np.sign(dm))))
        print(f'  vol={r} n={sel.sum()} corr(sign, past mid change):',' '.join(f'{h}s:{c:+.3f}/{s:+.3f}' for h,(c,s) in zip(H,out)))
    # sign autocorr across consecutive takers
    s=sg; print('  taker sign acf lag1..3', [round(np.corrcoef(s[l:],s[:-l])[0,1],4) for l in (1,2,3)])
