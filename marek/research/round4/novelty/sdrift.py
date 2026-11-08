"""On exact S: impact regression (0.5 s bins), residual VR/ACF, raw S VR/ACF, past->future betas. r3 vs r4."""
import numpy as np, pandas as pd
from common import *
B=0.5; K=120
def design(q,K):
    n=len(q); X=np.zeros((n,K))
    for j in range(K): X[j:,j]=q[:n-j]
    return X
for n in ['round3','round4']:
    o=L(n,'orders'); k=o[o.ioc&(o.filled>0)]; v=regime_1s(n)
    x=L(n,'strue').sp.to_numpy()
    nb=int(6000/B); xs=x[(np.arange(nb+1)*B*10).astype(int).clip(0,len(x)-1)]; dS=np.diff(xs)
    sg=np.where(k.side==0,1,-1)*k.filled**0.62; bi=(k.t/B).astype(int).clip(0,nb-1); vr=v[k.t.astype(int)]
    X=np.hstack([design(np.bincount(bi[~vr],weights=sg[~vr],minlength=nb),K),design(np.bincount(bi[vr],weights=sg[vr],minlength=nb),K)])
    tb=np.arange(nb)*B; bad=tb<70
    for on in shocks(n).onset: bad|=(tb>on-1)&(tb<on+10)
    ok=~bad; beta,*_=np.linalg.lstsq(X[ok],dS[ok],rcond=None); res=np.where(ok,dS-X@beta,0.0)
    np.save(f'cache/{n}_sresid05.npy',np.where(ok,res,np.nan))
    G=np.cumsum(beta[:K]); Gv=np.cumsum(beta[K:]); ii=[0,3,9,19,59,119]
    print(f'== {n} R2={1-res[ok].var()/dS[ok].var():.3f}  G at 0.5/2/5/10/30/60 s calm',G[ii].round(4).tolist(),' vol',Gv[ii].round(4).tolist())
    vb=v[tb.astype(int)]; cb=np.cumsum(~ok)
    for nm,ser in [('resid',res),('rawS',np.where(ok,dS,0.0))]:
        cs=np.cumsum(ser)
        for reg in [False,True]:
            out=[]
            for h in [1,2,4,10,20,40,80]:
                a=np.arange(0,nb-h); seg=(cb[a+h]-cb[a]==0)&(vb[a]==reg)&(vb[a+h]==reg)&ok[a]
                d=cs[a+h][seg]-cs[a][seg]; out.append(np.mean(d**2)/h)
            out=np.array(out)/out[0]
            bet=[]
            for P,H in [(2,2),(4,4),(10,10),(20,20),(4,20),(40,40)]:
                a=np.arange(P,nb-H); a=a[(cb[a+H]-cb[a-P]==0)&(vb[a]==reg)&(vb[a+H]==reg)&(vb[a-P]==reg)]
                p=cs[a]-cs[a-P]; f=cs[a+H]-cs[a]; bet.append(np.sum(p*f)/np.sum(p*p))
            print(f"  {nm:5} {'vol ' if reg else 'calm'} VR 1/2/5/10/20/40s:",out[1:].round(2).tolist(),' beta past->fut (1,1)(2,2)(5,5)(10,10)(2,10)(20,20)s:',np.round(bet,3).tolist())
