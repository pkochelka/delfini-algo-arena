import numpy as np, pandas as pd
from common import *
B=0.5
for n in ['round3','round4']:
    r=np.load(f'cache/{n}_sresid05.npy'); v=regime_1s(n); nb=len(r); vb=v[(np.arange(nb)*B).astype(int)]
    for reg in [False,True]:
        m=~np.isnan(r)&(vb==reg); x=np.where(m,r,0.0); mu=0
        ac=[]
        for l in [1,2,3,4,6,8,10,14,20,30,40,60]:
            mm=m[l:]&m[:-l]; ac.append(np.sum(x[l:][mm]*x[:-l][mm])/np.sqrt(np.sum(x[l:][mm]**2)*np.sum(x[:-l][mm]**2)))
        print(n,'vol ' if reg else 'calm','resid incr ACF at lag 0.5,1,1.5,2,3,4,5,7,10,15,20,30 s:',np.round(ac,3).tolist(),' sd',x[m].std().round(3))
