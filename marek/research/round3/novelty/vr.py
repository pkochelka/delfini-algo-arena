import numpy as np, pandas as pd
from common import *
def segid(v):
    d=np.r_[True, v[1:]!=v[:-1]]; return np.cumsum(d)
for n in ['round2','round3']:
    g=L(n,'grid'); v=regime_1s(n); sh=shocks(n)
    mid=g.mid.ffill().bfill().to_numpy(); t=g.t.to_numpy()
    # mark shocks as their own segments (exclude)
    bad=np.zeros(6001,bool)
    for x in sh.onset: bad[int(x)-1:int(x)+30]=True
    lab=np.where(bad,2,v.astype(int)); sid=segid(lab)
    s10=sid[t.astype(int)]; l10=lab[t.astype(int)]
    print(n)
    base={}
    for h in [1,2,5,10,20,30,45,60,90]:
        st=h*10; a=np.arange(len(mid)-st); b=a+st
        same=(s10[a]==s10[b])
        out=[]
        for reg in [0,1]:
            ok=same&(l10[a]==reg)
            d=mid[b[ok]]-mid[a[ok]]; var=np.mean(d**2)
            if h==1: base[reg]=var
            out.append(f'reg={reg} sd={np.sqrt(var):6.2f} VR={var/h/base[reg]:5.2f} n={ok.sum()//st}')
        print(f'  h={h:3}s', ' | '.join(out))
