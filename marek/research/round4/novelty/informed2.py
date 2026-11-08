"""Does taker imbalance predict the FUTURE non-impact residual beyond the past residual? (informed about drift?)"""
import numpy as np, pandas as pd
from common import *
B=0.5
for n in ['round3','round4']:
    r=np.load(f'cache/{n}_resid05.npy'); ok=~np.isnan(r); r0=np.nan_to_num(r); cs=np.cumsum(r0); nb=len(r)
    o=L(n,'orders'); k=o[o.ioc]
    if n=='round4': k=k[~L('round4','takers_tw').tw.to_numpy()]
    v=regime_1s(n); vb=v[(np.arange(nb)*B).astype(int)]
    sg=np.where(k.side==0,1,-1); bi=(k.t/B).astype(int).clip(0,nb-1)
    q=np.cumsum(np.bincount(bi,weights=sg,minlength=nb))
    cb=np.cumsum(~ok)
    for reg in [False,True]:
        for P,H in [(4,10),(4,20),(10,20),(2,4)]:
            a=np.arange(P,nb-H); a=a[(cb[a+H]-cb[a-P]==0)&(vb[a]==reg)]
            past=cs[a]-cs[a-P]; imb=q[a]-q[a-P]; fut=cs[a+H]-cs[a]
            X=np.column_stack([np.ones(len(a)),past/past.std(),imb/imb.std()])
            b,*_=np.linalg.lstsq(X,fut,rcond=None)
            # t-stat approx
            res=fut-X@b; se=np.sqrt(np.diag(np.linalg.inv(X.T@X))*res.var())
            print(f"{n} {'vol ' if reg else 'calm'} past {P*B}s -> fut {H*B}s: coef past_resid={b[1]:+.3f} (t {b[1]/se[1]:+.1f}), past_imb={b[2]:+.3f} (t {b[2]/se[2]:+.1f}) [ticks per sd], n={len(a)}")
