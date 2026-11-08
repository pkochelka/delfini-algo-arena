"""Event study of S around filled takers, split by realized kick residual (exact S, fine grid)."""
import numpy as np, pandas as pd
from common import *
for n in ['round2','round3']:
    o=L(n,'orders').reset_index(drop=True); s=pd.read_parquet(f'cache/{n}_s_at_insert.parquet'); S=s.s.to_numpy(); t=o.t.to_numpy(); v=regime_1s(n)
    mk=~o.ioc.to_numpy(); idx_mk=np.where(mk)[0]; tm=t[idx_mk]; Sm=S[idx_mk]
    def at(tt): return Sm[np.clip(np.searchsorted(tm,tt,side='right')-1,0,len(Sm)-1)]
    k=o[o.ioc&(o.filled>0)]; ki=k.index.to_numpy()
    p=idx_mk[np.searchsorted(idx_mk,ki)-1]; q=idx_mk[np.minimum(np.searchsorted(idx_mk,ki),len(idx_mk)-1)]
    kick=S[q]-S[p]; sg=np.where(k.side==0,1,-1); f=k.filled.to_numpy()**0.62; vr=v[k.t.astype(int)]
    bad=np.zeros(len(k),bool)
    for on in shocks(n).onset: bad|=(k.t.to_numpy()>on-62)&(k.t.to_numpy()<on+8)
    sel=(~vr)&~bad&((t[q]-t[p])<0.05)
    c=np.sum(kick[sel]*sg[sel]*f[sel])/np.sum(f[sel]**2); res=(kick-c*sg*f)*sg
    taus=[-1,-0.2,-0.05,0.05,0.2,0.5,1,2,5,10]
    t0=k.t.to_numpy()
    for lab,mask in [('res low',sel&(res<np.percentile(res[sel],20))),('res mid',sel&(np.abs(res)<np.percentile(np.abs(res[sel]),30))),('res high',sel&(res>np.percentile(res[sel],80)))]:
        base=at(t0[mask]-0.001)
        prof=[np.mean(sg[mask]*(at(t0[mask]+u)-base)) for u in taus]
        print(f'{n} calm {lab:8s} n={mask.sum()} kick={np.mean(sg[mask]*kick[mask]):+.3f}', ' '.join(f'{u}:{x:+.3f}' for u,x in zip(taus,prof)))
