"""Exact-dt scan between same-size orders at long range (up to 10 s takers, 2 s makers): looking for TWAP-like clocks."""
import numpy as np, pandas as pd
from common import *
for n in ['round3','round4']:
    o=L(n,'orders'); o=o.assign(tus=np.rint(o.t*1e6).astype(np.int64))
    for nm,sel,W in [('taker',o.ioc,10_000_000),('maker',~o.ioc,2_000_000)]:
        x=o[sel]; cnt={}
        for v,gg in x.groupby('vol'):
            if len(gg)<2: continue
            t=gg.tus.to_numpy()
            for l in range(1,min(len(t),60)):
                d=t[l:]-t[:-l]; d=d[d<=W]
                if len(d)==0: break
                u,c=np.unique(d,return_counts=True)
                for a,b in zip(u[c>=2],c[c>=2]): cnt[(a)]=cnt.get(a,0)+b
        s=pd.Series(cnt).sort_values(ascending=False)
        print(n,nm,'top exact same-size dt (us):',s.head(8).to_dict())
