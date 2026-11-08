"""Past->future regression of price on 0.1s grid, whole tape minus shock windows, 60s-block bootstrap. Optional regime filter."""
import numpy as np, pandas as pd, sys
from common import *
rng=np.random.default_rng(1)
def run(name,x,v,bad,label):
    print(label)
    for P,H in [(1,1),(2,2),(5,5),(10,10),(20,20),(30,30),(60,60),(5,30)]:
        a=np.arange(P*10,len(x)-H*10,5)
        okb=~(bad[a-P*10]|bad[a+H*10]|bad[a])
        out=[]
        for reg in ([None] if v is None else [0,1]):
            ok=okb if reg is None else okb&(v[a]==reg)&(v[a-P*10]==reg)&(v[a+H*10]==reg)
            aa=a[ok]; past=x[aa]-x[aa-P*10]; fut=x[aa+H*10]-x[aa]
            beta=np.sum(past*fut)/np.sum(past**2)
            blk=(aa//600); ub=np.unique(blk); groups=[np.where(blk==u)[0] for u in ub]; bs=[]
            for _ in range(200):
                idx=np.concatenate([groups[j] for j in rng.integers(0,len(groups),len(groups))]); bs.append(np.sum(past[idx]*fut[idx])/np.sum(past[idx]**2))
            out.append(f'reg={reg} beta={beta:+.3f}±{np.std(bs):.3f}')
        print(f'  {P:2}s->{H:2}s', ' | '.join(out))
# round1 S control
d=np.load('/Users/marek/Documents/delfini-algo-arena/data/.cache/round1_S.npz'); t=(d['t']-d['t'][0])/1e6
T=t[-1]; grid=np.arange(0,T,0.1); x=d['S'][np.clip(np.searchsorted(t,grid,side='right')-1,0,None)]
run('r1',x,None,np.zeros(len(grid),bool),'round1 true S (whole tape)')
for n,src in [('round2','strue'),('round2','sproxy'),('round3','sproxy')]:
    g=L(n,src); x=g.sp.to_numpy(); t=g.t.to_numpy(); v1=regime_1s(n); v=v1[t.astype(int)].astype(int)
    bad=np.zeros(len(t),bool)
    for o in shocks(n).onset: bad|=(t>o-1)&(t<o+30)
    run(n,x,v,bad,f'{n} {src}')
