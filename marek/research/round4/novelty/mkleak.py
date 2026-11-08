"""Do regular maker quotes lean toward the future S move (makers know the drift)? price - S(t) vs S(t+h) - S(t)."""
import numpy as np, pandas as pd
from common import *
for n in ['round3','round4']:
    o=L(n,'orders'); s=pd.read_parquet(f'cache/{n}_s_at_insert.parquet').s.to_numpy(); x=L(n,'strue').sp.to_numpy()
    S=lambda t: np.interp(t,np.arange(len(x))*0.1,x)
    ch=pd.read_parquet(f'cache/{n}_makers_chains.parquet')[['oid','chain']]
    o['s']=s; m=o[~o.ioc].merge(ch,on='oid',how='left'); m=m[~m.chain&(m.vol!=15)]
    m=m.sample(300000,random_state=0)
    v=regime_1s(n); vr=v[m.t.astype(int)]
    bad=np.zeros(len(m),bool)
    for a in shocks(n).onset: bad|=((m.t>a-1)&(m.t<a+10)).to_numpy()
    for R in [False,True]:
        for sd in [0,1]:
            mm=(vr==R)&(m.side==sd).to_numpy()&~bad&(m.t>70).to_numpy()
            off=(m.price-m.s).to_numpy()[mm]; t=m.t.to_numpy()[mm]
            out=[]
            for h in [-1,0.5,1,2]:
                f=(S(t+h)-S(t)) if h>0 else (S(t)-S(t+h))
                out.append(f'{h:+}s:{np.corrcoef(off,f)[0,1]:+.3f}')
            print(n,'vol ' if R else 'calm','bid' if sd==0 else 'ask','corr(price-S, S move [past -1s / future h])',' '.join(out))
