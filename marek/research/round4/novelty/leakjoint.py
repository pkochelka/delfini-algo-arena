"""Joint leak regression (r4 vs r3): future S move (s_hat t+H minus causal s_filt t) on causal features:
past s_filt moves (0.5 s, 0.5-2 s), mid - s_filt gap, taker imbalance 1 s, ladder burst-start imbalance 1 s, TWAP side, r15 agent side."""
import numpy as np, pandas as pd
from common import *
for n in ['round3','round4']:
    o=L(n,'orders'); p=pd.read_parquet(f'/Users/marek/Documents/delfini-algo-arena/marek/research/{n}/s_dynamics/s_path.parquet',columns=['s_hat','s_filt'])
    t=o.t.to_numpy(); grid=np.arange(0,6000,0.1); N=len(grid); i=np.clip(np.searchsorted(t,grid,side='right')-1,0,len(t)-1)
    sf=p.s_filt.to_numpy()[i]; sh=p.s_hat.to_numpy()[i]; mid=L(n,'grid').mid.ffill().bfill().to_numpy()
    k=o[o.ioc]; sg=np.where(k.side==0,1,-1); q=np.cumsum(np.bincount((k.t*10).astype(int).clip(0,N-1),weights=sg,minlength=N))
    ch=pd.read_parquet(f'cache/{n}_makers_chains.parquet'); st=ch[ch.chain&(ch.cidx==0)]
    lq=np.cumsum(np.bincount((st.t*10).astype(int).clip(0,N-1),weights=np.where(st.side==0,1,-1),minlength=N))
    tw=np.zeros(N); ag=np.zeros(N)
    if n=='round4':
        e=pd.read_csv('twap_episodes.csv')
        for _,r in e.iterrows(): tw[int(r.t0*10)+6:int(r.t1*10)+1]=1 if r.side=='B' else -1
        d=pd.read_pickle('../makers/cache/r15_v2.pkl'); d=d[d.L>=2]
        for c,g in d.groupby('chain'): ag[int(g.t_us.min()/1e5)+1:int(g.end_t_us.max()/1e5)+1]=-g.sg.iloc[0]  # sign = direction of S drift
    v=regime_1s(n); bad=grid<70
    for on in shocks(n).onset: bad|=(grid>on-1)&(grid<on+12)
    a=np.arange(200,N-400,5); a=a[~bad[a]&~bad[a+300]&~bad[a-100]]; reg=v[grid[a].astype(int)]
    print('=====',n)
    for R in [False,True]:
        aa=a[reg==R]
        F={'dS0.5':sf[aa]-sf[aa-5],'dS2':sf[aa-5]-sf[aa-20],'S-mid':sf[aa]-mid[aa],'timb1':q[aa]-q[aa-10],'ladimb1':lq[aa]-lq[aa-10],'twap':tw[aa],'r15':ag[aa]}
        names=[k_ for k_,x in F.items() if x.std()>0]; X=np.column_stack([F[k_]/F[k_].std() for k_ in names]); X=np.c_[np.ones(len(aa)),X]
        for H in [2,5]:
            f=sh[aa+H*10]-sf[aa]; b,*_=np.linalg.lstsq(X,f,rcond=None); res=f-X@b
            se=np.sqrt(np.diag(np.linalg.inv(X.T@X))*res.var()); R2=1-res.var()/f.var()
            print(f" {'vol ' if R else 'calm'} H={H}s R2={R2:.3f} coef (ticks per sd) [t]:",' '.join(f'{nm}={b[j+1]:+.2f}[{b[j+1]/se[j+1]:+.0f}]' for j,nm in enumerate(names)))
