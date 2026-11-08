import numpy as np, pandas as pd
from common import *
def segid(v):
    d=np.r_[True, v[1:]!=v[:-1]]; return np.cumsum(d)
rng=np.random.default_rng(0)
for n in ['round2','round3']:
    g=L(n,'sproxy'); v=regime_1s(n); sh=shocks(n)
    x=g.sp.to_numpy(); t=g.t.to_numpy()
    bad=np.zeros(6001,bool)
    for o in sh.onset: bad[int(o)-1:int(o)+30]=True
    lab=np.where(bad,2,v.astype(int)); sid=segid(lab); s10=sid[t.astype(int)]; l10=lab[t.astype(int)]
    print(n)
    for P,H in [(1,1),(2,2),(5,5),(10,10),(20,20),(30,30),(5,30),(30,5)]:
        a=np.arange(P*10,len(x)-H*10,5)  # every 0.5s
        out=[]
        for reg in [0,1]:
            ok=(s10[a-P*10]==s10[a+H*10])&(l10[a]==reg)
            aa=a[ok]; past=x[aa]-x[aa-P*10]; fut=x[aa+H*10]-x[aa]
            beta=np.sum(past*fut)/np.sum(past**2)
            # block bootstrap by segment
            segs=s10[aa]; us=np.unique(segs); bs=[]
            for _ in range(200):
                pick=rng.choice(us,len(us)); idx=np.concatenate([np.where(segs==u)[0] for u in pick]) if len(us)<400 else None
                if idx is None: break
                bs.append(np.sum(past[idx]*fut[idx])/np.sum(past[idx]**2))
            out.append(f'reg={reg} beta={beta:+.3f}±{np.std(bs):.3f} segs={len(us)}')
        print(f'  past {P:2}s -> fut {H:2}s:', ' | '.join(out))
