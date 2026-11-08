"""Residual (non-impact) mid drift: persistence time and per-segment structure."""
import numpy as np, pandas as pd
from common import *
B=0.5; K=60
def design(q,K):
    n=len(q); X=np.zeros((n,K))
    for j in range(K): X[j:,j]=q[:n-j]
    return X
for n in ['round3','round4']:
    o=L(n,'orders'); k=o[o.ioc&(o.filled>0)]; v=regime_1s(n)
    g=L(n,'grid'); mid=g.mid.ffill().bfill().to_numpy()
    nb=int(6000/B); xs=mid[(np.arange(nb+1)*B*10).astype(int).clip(0,len(mid)-1)]; dS=np.diff(xs)
    sg=np.where(k.side==0,1,-1)*k.filled**0.62; bi=(k.t/B).astype(int).clip(0,nb-1); vr=v[k.t.astype(int)]
    X=np.hstack([design(np.bincount(bi[~vr],weights=sg[~vr],minlength=nb),K),design(np.bincount(bi[vr],weights=sg[vr],minlength=nb),K)])
    tb=np.arange(nb)*B; bad=tb<40
    for on in shocks(n).onset: bad|=(tb>on-1)&(tb<on+10)
    ok=~bad; beta,*_=np.linalg.lstsq(X[ok],dS[ok],rcond=None); res=np.where(ok,dS-X@beta,0.0)
    np.save(f'cache/{n}_resid05.npy',np.where(ok,res,np.nan))
    vb=v[tb.astype(int)]
    # smoothed residual: 5s moving sum, autocorr at lags
    for reg in [False,True]:
        m=ok&(vb==reg); r=np.where(m,res,0.0); c=np.convolve(r,np.ones(10),'same')
        lags=[10,20,40,60,120]; ac=[]
        for l in lags:
            mm=m[l:]&m[:-l]; ac.append(np.corrcoef(c[l:][mm],c[:-l][mm])[0,1])
        print(n,'calm' if not reg else 'vol ','5s-sum residual corr at lag 5/10/20/30/60 s:',np.round(ac,3).tolist(),' resid sd per 0.5s',r[m].std().round(3))
    # variance ratio of cumulative residual within same-regime stretches
    cs=np.cumsum(res)
    for reg in [False,True]:
        out=[]
        for h in [1,2,4,10,20,40,80]:
            a=np.arange(0,nb-h); seg=ok[a]&ok[a+h]&(vb[a]==reg)&(vb[a+h]==reg)
            # require all ok between: approximate via cumulative bad count
            cb=np.cumsum(~ok); seg&=(cb[a+h]-cb[a])==0
            d=cs[a+h][seg]-cs[a][seg]; out.append(np.mean(d**2)/h)
        out=np.array(out)/out[0]; print('   VR of residual', 'calm' if not reg else 'vol', dict(zip([0.5,1,2,5,10,20,40],out.round(2))))
