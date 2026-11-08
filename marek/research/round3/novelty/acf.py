import numpy as np, pandas as pd
from common import *
for n in ['round2','round3']:
    g=L(n,'grid'); v=regime_1s(n); sh=shocks(n)
    mid=g.mid.ffill().bfill().to_numpy(); t=g.t.to_numpy()
    bad=np.zeros(len(t),bool)
    for x in sh.onset: bad|=(t>x-1)&(t<x+30)
    bad|=t<5
    print(n)
    base={}
    for h in [0.1,0.5,1,2,5,10,20,30,60]:
        st=int(round(h*10)); N=len(mid)-2*st
        r1=mid[st:st+N]-mid[:N]; r2=mid[2*st:2*st+N]-mid[st:st+N]
        vv=v[t[:N].astype(int)]; vv2=v[np.minimum(t[2*st:2*st+N].astype(int),6000)]
        bb=bad[:N]|bad[st:st+N]|bad[2*st:2*st+N]
        out=[]
        for reg in [False,True]:
            ok=(~bb)&(vv==reg)&(vv2==reg)
            sd=r1[ok].std(); 
            if h==1: base[reg]=sd
            vr=(sd**2/h)/(base.get(reg,np.nan)**2) if h>=1 else np.nan
            out.append(f'vol={reg}: sd={sd:6.3f} VR(vs1s)={vr:5.2f} acf={np.corrcoef(r1[ok],r2[ok])[0,1]:+.3f}')
        print(f'  h={h:5}s', ' | '.join(out))
