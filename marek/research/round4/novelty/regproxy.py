"""Crude 2-state Poisson HMM on 1 s counts of (2ms-chain starts, takers, maker inserts 20-70 lots) -> volatile mask per second."""
import sys, numpy as np, pandas as pd
from common import L, regime_1s
def build(n):
    o=L(n,'orders'); ch=pd.read_parquet(f'cache/{n}_makers_chains.parquet')
    st=ch[ch.chain&(ch.cidx==0)].t.to_numpy()
    k=o[o.ioc].t.to_numpy()
    X=np.stack([np.bincount(st.astype(int),minlength=6000)[:6000],np.bincount(k.astype(int),minlength=6000)[:6000]],1).astype(float)
    # init
    lam=np.array([[1.2,8.0],[10.,12.]]); P=np.array([[0.993,0.007],[0.017,0.983]]); pi=np.array([.7,.3])
    from scipy.stats import poisson
    for it in range(30):
        lp=np.stack([poisson.logpmf(X,lam[s]).sum(1) for s in range(2)],1)
        T=len(X); a=np.zeros((T,2)); c=np.zeros(T); e=np.exp(lp-lp.max(1,keepdims=True))
        a[0]=pi*e[0]; c[0]=a[0].sum(); a[0]/=c[0]
        for t in range(1,T): a[t]=(a[t-1]@P)*e[t]; c[t]=a[t].sum(); a[t]/=c[t]
        b=np.ones((T,2))
        for t in range(T-2,-1,-1): b[t]=P@(e[t+1]*b[t+1]); b[t]/=c[t+1]
        g=a*b; g/=g.sum(1,keepdims=True)
        xi=np.zeros((2,2))
        for t in range(T-1): m=np.outer(a[t],e[t+1]*b[t+1])*P; xi+=m/m.sum()
        lam=(g.T@X)/g.sum(0)[:,None]
    v=g[:,1]>0.5
    print(n,'lam',lam.round(2).tolist(),'P',P.round(4).tolist(),'vol frac',v.mean().round(3))
    return v
if __name__=='__main__':
    for n in sys.argv[1:]:
        v=build(n); np.save(f'cache/{n}_volproxy.npy',v)
        if n=='round3':
            vt=regime_1s('round3',force_flow=True)[:6000]; print(' agreement with flow regimes',(v==vt).mean())
