"""Causal leak: future S (smoothed, target) vs past move in causal s_filt and in mid. r3 vs r4, by regime."""
import numpy as np, pandas as pd
from common import *
for n in ['round3','round4']:
    o=L(n,'orders'); p=pd.read_parquet(f'/Users/marek/Documents/delfini-algo-arena/marek/research/{n}/s_dynamics/s_path.parquet',columns=['s_hat','s_filt'])
    t=o.t.to_numpy(); grid=np.arange(0,6000,0.1); i=np.clip(np.searchsorted(t,grid,side='right')-1,0,len(t)-1)
    sf=p.s_filt.to_numpy()[i]; sh=p.s_hat.to_numpy()[i]; mid=L(n,'grid').mid.ffill().bfill().to_numpy()
    v=regime_1s(n); N=len(grid); bad=grid<70
    for on in shocks(n).onset: bad|=(grid>on-1)&(grid<on+12)
    a=np.arange(200,N-400,5); a=a[~bad[a]&~bad[a+300]&~bad[a-100]]; reg=v[grid[a].astype(int)]
    print('=====',n)
    for R in [False,True]:
        aa=a[reg==R]; out=[]
        for P in [0.5,1,2]:
            for H in [1,2,5,10,30]:
                p_=sf[aa]-sf[aa-int(P*10)]; f=sh[aa+H*10]-sf[aa]
                pm=mid[aa]-mid[aa-int(P*10)]; fm=mid[aa+H*10]-mid[aa]
                out.append((P,H,np.corrcoef(p_,f)[0,1],np.sum(p_*f)/np.sum(p_*p_),np.corrcoef(pm,fm)[0,1]))
        print(' vol' if R else ' calm'); print('   P   H  corr(sfilt past, future S-sfilt)  beta   corr(mid past, mid fut)')
        for P,H,c,b,cm in out: print(f'  {P:3} {H:3}   {c:+.3f}   {b:+.3f}   {cm:+.3f}')
