"""Shape of trade-free S motion: histogram of S velocity in 0.3 s no-fill windows (calm, vol), r3 vs r4."""
import numpy as np, pandas as pd
from common import *
B=0.1
for n in ['round3','round4']:
    x=L(n,'strue').sp.to_numpy(); v=regime_1s(n); o=L(n,'orders'); k=o[o.ioc&(o.filled>0)]
    nb=len(x)-1; fill=np.bincount((k.t/B).astype(int).clip(0,nb-1),minlength=nb)>0
    t=np.arange(nb)*B; bad=t<70
    for on in shocks(n).onset: bad|=(t>on-1)&(t<on+10)
    vb=v[t.astype(int)]
    for reg in [False,True]:
        nf=(~fill)&(~bad)&(vb==reg); W=3
        run=np.convolve(nf.astype(int),np.ones(W,int),'valid')==W; i=np.where(run)[0][::W]
        vel=(x[i+W]-x[i])/(W*B)
        q=np.percentile(np.abs(vel),[25,50,75,90,95,99]).round(2)
        h=np.histogram(np.clip(vel,-4,4),bins=16,range=(-4,4))[0]
        print(f"{n} {'vol ' if reg else 'calm'} windows={len(i)} |vel| q25/50/75/90/95/99 (tick/s)={q.tolist()} kurt={pd.Series(vel).kurt():.1f}")
        print('     hist -4..4 by 0.5:',h.tolist())
