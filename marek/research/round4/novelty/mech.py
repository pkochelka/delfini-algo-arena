"""Is r4 momentum flow-driven (takers chase moves -> kicks continue) or S's own drift?
(1) distributed-lag impact regression of mid change on signed filled^0.62 (0.5 s bins), then ACF of the residual.
(2) taker sign ~ past mid move + past taker signs (linear prob)."""
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
    qc=np.bincount(bi[~vr],weights=sg[~vr],minlength=nb); qv=np.bincount(bi[vr],weights=sg[vr],minlength=nb)
    X=np.hstack([design(qc,K),design(qv,K)])
    tb=np.arange(nb)*B; bad=tb<40
    for on in shocks(n).onset: bad|=(tb>on-1)&(tb<on+10)
    ok=~bad
    beta,*_=np.linalg.lstsq(X[ok],dS[ok],rcond=None)
    res=np.where(ok,dS-X@beta,np.nan)
    R2=1-np.nanvar(res)/np.var(dS[ok])
    vb=v[tb.astype(int)]
    out=[]
    for reg in [False,True]:
        m=ok&(vb==reg); r=np.where(m,res,np.nan)
        ac=[np.nanmean(r[l:]*r[:-l])/np.nanmean(r*r) for l in (1,2,4,10)]
        # sum of residual over 2s -> next 2s
        cs=np.cumsum(np.nan_to_num(r)); a=np.arange(8,nb-8); a=a[m[a]&m[a-4]&m[a+4]]
        p=cs[a]-cs[a-4]; f=cs[a+4]-cs[a]; b2=np.sum(p*f)/np.sum(p*p)
        a=np.arange(20,nb-20); a=a[m[a]&m[a-10]&m[a+10]]; p=cs[a]-cs[a-10]; f=cs[a+20]-cs[a]; b5=np.sum(p*f)/np.sum(p*p)
        out.append(f"{'vol' if reg else 'calm'}: resid acf lag0.5/1/2/5s={np.round(ac,3).tolist()} beta 2s->2s={b2:+.3f} 5s->10s={b5:+.3f}")
    G=np.cumsum(beta[:K]); Gv=np.cumsum(beta[K:])
    print(n,'R2',round(R2,3),'| calm G(0.5,2,5,10,30s)',G[[0,3,9,19,59]].round(4).tolist(),'| vol G',Gv[[0,3,9,19,59]].round(4).tolist())
    for s in out: print('   ',s)
    # (2) taker sign regression
    kk=o[o.ioc].copy(); s=np.where(kk.side==0,1,-1).astype(float)
    top=np.load(f'cache/{n}_top.npz'); tt=top['t']; b_=top['bid'].astype(float); a_=top['ask'].astype(float); b_[b_<=0]=np.nan; a_[a_<=0]=np.nan
    mm=pd.Series((a_+b_)/2).ffill().bfill().to_numpy()
    def midat(x): return mm[np.clip(np.searchsorted(tt,x,side='right')-1,0,len(mm)-1)]
    m0=mm[kk.row.to_numpy()-1]
    feats=[m0-midat(kk.t.to_numpy()-0.5), midat(kk.t.to_numpy()-0.5)-midat(kk.t.to_numpy()-2)]
    for l in range(1,6): feats.append(np.r_[np.zeros(l),s[:-l]])
    F=np.column_stack(feats); vk=v[kk.t.astype(int)]
    bad=np.zeros(len(kk),bool)
    for on in shocks(n).onset: bad|=(kk.t>on-1)&(kk.t<on+10)
    if n=='round4':
        tw=L('round4','takers_tw').tw.to_numpy(); bad|=tw
    for reg in [False,True]:
        m=(vk==reg)&~bad&(kk.t>30).to_numpy()
        Fz=F[m]/F[m].std(0); bb,*_=np.linalg.lstsq(np.column_stack([np.ones(m.sum()),Fz]),s[m],rcond=None)
        print(f"   sign reg {'vol' if reg else 'calm'} (std coefs): dmid(0.5s)={bb[1]:+.3f} dmid(2..0.5s)={bb[2]:+.3f} prev signs 1..5={np.round(bb[3:],3).tolist()}")
