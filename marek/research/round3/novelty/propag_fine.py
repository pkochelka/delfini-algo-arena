import numpy as np, pandas as pd
from common import *
B=0.1; K=50
def design(q, K):
    n=len(q); X=np.zeros((n,K),np.float32)
    for k in range(K): X[k:,k]=q[:n-k]
    return X
for n,src in [('round2','strue'),('round3','strue'),('round2','sproxy'),('round3','sproxy')]:
    o=L(n,'orders'); k=o[o.ioc&(o.filled>0)]; v=regime_1s(n)
    x=L(n,src).sp.to_numpy(); nb=len(x)-1; dS=np.diff(x)
    sg=np.where(k.side==0,1,-1)*k.filled**0.62
    bi=(k.t/B).astype(int).clip(0,nb-1); vr=v[k.t.astype(int)]
    qc=np.bincount(bi[~vr],weights=sg[~vr],minlength=nb); qv=np.bincount(bi[vr],weights=sg[vr],minlength=nb)
    X=np.hstack([design(qc,K),design(qv,K)])
    bad=np.zeros(nb,bool); tb=np.arange(nb)*B
    for on in shocks(n).onset: bad|=(tb>on-1)&(tb<on+8)
    ok=~bad; ok[:K]=False
    beta,*_=np.linalg.lstsq(X[ok],dS[ok],rcond=None)
    for name,sl in [('calm',slice(0,K)),('vol',slice(K,2*K))]:
        G=np.cumsum(beta[sl]); pts=[0,1,2,4,7,9,14,19,29,49]
        print(n,src,name,' '.join(f'{(j+1)*B:.1f}:{G[j]/G[2]:.3f}' for j in pts), f' (G0.3={G[2]:.4f})')
