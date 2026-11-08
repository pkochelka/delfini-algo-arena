import numpy as np, pandas as pd
from common import *
for n,src in [('round2','strue'),('round2','sproxy'),('round3','sproxy')]:
    g=L(n,src); x=g.sp.to_numpy(); sh=shocks(n)
    def at(t): return x[np.clip((np.asarray(t)*10).astype(int),0,len(x)-1)]
    on=sh.onset.to_numpy(); pre=at(on-0.5); post=at(on+4.0)
    J=post-pre
    rows=[]
    for h in [2,5,10,20,30,60,120]:
        d=at(on+4+h)-post; b=np.sum(J*d)/np.sum(J*J); rows.append(f'{h}s:{b:+.3f}')
    print(n,src,'J sd',J.std().round(1),'| beta(future move on jump):',' '.join(rows))
    # also pre-shock leak: move in 30s before onset vs J
    pl=pre-at(on-30); print('   pre-30s move corr with J', np.corrcoef(pl,J)[0,1].round(3), ' J values', np.round(J).astype(int).tolist())
