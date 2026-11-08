"""Which S time do taker limits reference? band fraction of sign*(limit - S(t+delta)) for delta in seconds."""
import numpy as np, pandas as pd
from common import *
for n in ['round3','round4']:
    o=L(n,'orders'); x=L(n,'strue').sp.to_numpy(); v=regime_1s(n)
    k=o[o.ioc].copy()
    if n=='round4': k=k[~L('round4','takers_tw').tw.to_numpy()]
    bad=np.zeros(len(k),bool)
    for a in shocks(n).onset: bad|=(k.t>a-1)&(k.t<a+10)
    k=k[~bad&(k.t>5)&(k.t<5990)]; sg=np.where(k.side==0,1,-1); vr=v[k.t.astype(int)]
    def S(t): return np.interp(t,np.arange(len(x))*0.1+0.05,x)
    for r in [False,True]:
        band=1 if not r else 2.55; out=[]
        for d in [-1,-0.5,-0.2,-0.1,0,0.1,0.2,0.5,1]:
            g=sg[vr==r]*(k.price.to_numpy()[vr==r]-S(k.t.to_numpy()[vr==r]+d))
            out.append(f'{d:+.1f}:{np.mean(np.abs(g-band)<=0.5):.3f}')
        print(n,'vol ' if r else 'calm','frac |g-band|<=0.5 by delta(s):',' '.join(out))
