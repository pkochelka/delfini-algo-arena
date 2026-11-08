"""S motion in windows with no fills: r3 has no diffusion, so any motion there is drift/decay. Also: does it continue the prior move?"""
import numpy as np, pandas as pd
from common import *
B=0.1
for n in ['round3','round4']:
    x=L(n,'strue').sp.to_numpy(); v=regime_1s(n); o=L(n,'orders'); k=o[o.ioc&(o.filled>0)]
    nb=len(x)-1; d=np.diff(x)
    fill=np.bincount((k.t/B).astype(int).clip(0,nb-1),minlength=nb)>0
    # windows of >=5 consecutive no-fill bins (0.5 s)
    t=np.arange(nb)*B; bad=t<70
    for on in shocks(n).onset: bad|=(t>on-1)&(t<on+10)
    vb=v[t.astype(int)]
    cs=np.cumsum(np.r_[0,d])
    for reg in [False,True]:
        # find bins i where bins i..i+4 no fill and not bad
        nf=(~fill)&(~bad)&(vb==reg)
        run=np.convolve(nf.astype(int),np.ones(5,int),'valid')==5
        i=np.where(run)[0]; i=i[i>20]
        mv=cs[i+5]-cs[i]   # S move over 0.5 s with no fills
        past=cs[i]-cs[i-20] # move over previous 2 s
        b=np.sum(mv*past)/np.sum(past**2)
        print(f"{n} {'vol ' if reg else 'calm'} no-fill 0.5s windows={len(i)} sd(S move)={mv.std():.3f} mean|move|={np.abs(mv).mean():.3f}  beta on past 2s move={b:+.4f} corr={np.corrcoef(mv,past)[0,1]:+.3f}")
