"""Is the residual S motion tied to fills (e.g. noise xi applied over a ramp) or to clock time?
corr(n_fills in bin i, resid^2 in bin i+l) for l=-4..12 (0.5 s bins), per regime. Also resid var per bin vs #fills."""
import numpy as np, pandas as pd
from common import *
B=0.5
for n in ['round3','round4']:
    r=np.load(f'cache/{n}_sresid05.npy'); nb=len(r); ok=~np.isnan(r); v=regime_1s(n); vb=v[(np.arange(nb)*B).astype(int)]
    o=L(n,'orders'); k=o[o.ioc&(o.filled>0)]
    nf=np.bincount((k.t/B).astype(int).clip(0,nb-1),minlength=nb).astype(float)
    for reg in [False,True]:
        m=ok&(vb==reg); r2=np.where(m,r**2,np.nan)
        out=[]
        for l in range(-4,13):
            a=np.arange(max(0,-l),min(nb,nb-l)); mm=m[a]&m[a+l]
            out.append(np.corrcoef(nf[a][mm],r2[a+l][mm])[0,1])
        # var of resid by fill count
        vv=[np.nanmean(r2[m&(nf==c)]) for c in range(0,8)]
        print(n,'vol ' if reg else 'calm','corr(nfill_i, resid^2_{i+l}) l=-4..12:',' '.join(f'{x:+.2f}' for x in out))
        print('      resid var by #fills in bin 0..7:',np.round(vv,3).tolist())
