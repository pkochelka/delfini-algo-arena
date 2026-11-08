"""Distributed-lag propagator: dS(bin) on lagged signed filled^0.62 of takers, separately for calm/volatile takers."""
import numpy as np, pandas as pd
from common import *
B=0.5; K=120  # 60 s of lags
def design(q, K):
    n=len(q); X=np.zeros((n,K))
    for k in range(K): X[k:,k]=q[:n-k]
    return X
for n,src in [('round2','strue'),('round3','sproxy')]:
    o=L(n,'orders'); k=o[o.ioc&(o.filled>0)]; v=regime_1s(n)
    g=L(n,src); x=g.sp.to_numpy()
    nb=int(6000/B); xs=x[(np.arange(nb+1)*B*10).astype(int).clip(0,len(x)-1)]; dS=np.diff(xs)
    sg=np.where(k.side==0,1,-1)*k.filled**0.62
    bi=(k.t/B).astype(int).clip(0,nb-1); vr=v[k.t.astype(int)]
    qc=np.bincount(bi[~vr],weights=sg[~vr],minlength=nb); qv=np.bincount(bi[vr],weights=sg[vr],minlength=nb)
    X=np.hstack([design(qc,K),design(qv,K)])
    bad=np.zeros(nb,bool); tb=np.arange(nb)*B
    for on in shocks(n).onset: bad|=(tb>on-1)&(tb<on+8)
    ok=~bad; ok[:K]=False
    beta,*_=np.linalg.lstsq(X[ok],dS[ok],rcond=None)
    res=dS[ok]-X[ok]@beta; s2=res.var(); cov=s2*np.linalg.inv(X[ok].T@X[ok])
    for name,sl in [('calm',slice(0,K)),('vol',slice(K,2*K))]:
        G=np.cumsum(beta[sl]); 
        # SE of cumulative
        C=cov[sl,sl]; se=np.sqrt([C[:j+1,:j+1].sum() for j in range(K)])
        pts=[0,1,3,5,9,19,39,59,89,119]
        print(n,name,' '.join(f'{(j+1)*B:g}s:{G[j]:.4f}±{se[j]:.4f}' for j in pts))
