"""Fine ACF of S increments on 0.25 s bins excluding bins with fills (to remove impact), r4."""
import numpy as np, pandas as pd
from common import *
B=0.25
for n in ['round3','round4']:
    x=L(n,'strue').sp.to_numpy(); v=regime_1s(n); o=L(n,'orders'); k=o[o.ioc&(o.filled>0)]
    t=np.arange(0,6000,B); xs=np.interp(t,np.arange(len(x))*0.1,x); d=np.diff(xs); nb=len(d)
    fill=np.bincount((k.t/B).astype(int).clip(0,nb-1),minlength=nb)>0
    bad=t[:-1]<70
    for on in shocks(n).onset: bad|=(t[:-1]>on-1)&(t[:-1]<on+10)
    vb=v[t[:-1].astype(int)]
    for reg in [False,True]:
        m=(~bad)&(~fill)&(vb==reg); xx=np.where(m,d,0.0)
        lags=np.arange(1,29)
        ac=[np.sum(xx[l:]*xx[:-l])/np.sqrt(np.sum(xx[l:]**2*m[:-l])*np.sum(xx[:-l]**2*m[l:])) for l in lags]
        print(n,'vol ' if reg else 'calm','no-fill bins',m.sum(),'ACF lags 0.25..7s:',' '.join(f'{a:+.2f}' for a in ac))
