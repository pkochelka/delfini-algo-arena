"""Trading check: transient-impact stock Y(t) predicts future S drift? (exact S, r3 vs r2 control)."""
import numpy as np, pandas as pd
from common import *
for n in ['round2','round3']:
    o=L(n,'orders'); k=o[o.ioc&(o.filled>0)]; v=regime_1s(n); x=L(n,'strue').sp.to_numpy()
    c=np.where(v[k.t.astype(int)],0.062,0.0166); T=np.where(v[k.t.astype(int)],18.0,35.0)
    kick=c*0.5*np.where(k.side==0,1,-1)*k.filled**0.62
    B=0.1; nb=len(x); q=np.bincount((k.t/B).astype(int).clip(0,nb-1),weights=kick,minlength=nb)
    # Y with regime-dependent decay
    vv=v[(np.arange(nb)*B).astype(int)]; dec=np.exp(-B/np.where(vv,18.0,35.0))
    Y=np.zeros(nb); acc=0.0
    for i in range(nb): acc=acc*dec[i]+q[i]; Y[i]=acc
    bad=np.zeros(nb,bool); tb=np.arange(nb)*B
    for on in shocks(n).onset: bad|=(tb>on-1)&(tb<on+40)
    for H in [5,10,30,60]:
        h=int(H/B); a=np.arange(0,nb-h,10); a=a[~bad[a]&~bad[a+h]]
        fut=x[a+h]-x[a]; b=np.sum(Y[a]*fut)/np.sum(Y[a]**2); r=np.corrcoef(Y[a],fut)[0,1]
        print(f'{n} H={H}s: slope of future dS on Y = {b:+.3f} (theory -(1-e^-H/T)), corr={r:+.3f}')
