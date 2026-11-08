"""Propagator with finer output + exp/power fits + residual (non-impact) reversion."""
import numpy as np, pandas as pd
from scipy.optimize import curve_fit
from common import *
B=0.5; K=240
def design(q, K):
    n=len(q); X=np.zeros((n,K))
    for k in range(K): X[k:,k]=q[:n-k]
    return X
n='round3'
o=L(n,'orders'); k=o[o.ioc&(o.filled>0)]; v=regime_1s(n)
x=L(n,'strue').sp.to_numpy()
nb=int(6000/B); xs=x[(np.arange(nb+1)*B*10).astype(int).clip(0,len(x)-1)]; dS=np.diff(xs)
sg=np.where(k.side==0,1,-1)*k.filled**0.62
bi=(k.t/B).astype(int).clip(0,nb-1); vr=v[k.t.astype(int)]
qc=np.bincount(bi[~vr],weights=sg[~vr],minlength=nb); qv=np.bincount(bi[vr],weights=sg[vr],minlength=nb)
X=np.hstack([design(qc,K),design(qv,K)])
bad=np.zeros(nb,bool); tb=np.arange(nb)*B
for on in shocks(n).onset: bad|=(tb>on-1)&(tb<on+8)
ok=~bad; ok[:K]=False
beta,*_=np.linalg.lstsq(X[ok],dS[ok],rcond=None)
tau=(np.arange(K)+1)*B
for name,sl in [('calm',slice(0,K)),('vol',slice(K,2*K))]:
    G=np.cumsum(beta[sl]); m=tau>=2
    f=lambda t,c,p,T: c*(p+(1-p)*np.exp(-t/T))
    pe,_=curve_fit(f,tau[m],G[m],p0=[G[3],0.5,15])
    fp=lambda t,c,p,a: c*(p+(1-p)*(1+t)**(-a))
    pp,_=curve_fit(fp,tau[m],G[m],p0=[G[3],0.3,0.5],maxfev=20000)
    re=np.sqrt(np.mean((G[m]-f(tau[m],*pe))**2)); rp=np.sqrt(np.mean((G[m]-fp(tau[m],*pp))**2))
    print(name,'exp fit c,p,T',pe.round(4),'rmse',re.round(5),'| power c,p,a',pp.round(4),'rmse',rp.round(5))
    print('  G at 90s,120s:',G[179].round(4),G[239].round(4))
# residual reversion
res=np.full(nb,np.nan); res[ok]=dS[ok]-X[ok]@beta
r=np.nan_to_num(res); cs=np.cumsum(r)
for P,H in [(2,2),(5,5),(10,10),(20,20),(5,30)]:
    p=int(P/B); h=int(H/B); a=np.arange(p,nb-h,2); a=a[ok[a]]
    past=cs[a]-cs[a-p]; fut=cs[a+h]-cs[a]
    print(f'residual {P}s->{H}s beta={np.sum(past*fut)/np.sum(past**2):+.3f}')
