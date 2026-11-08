"""Per-taker S kick decomposition: does the noise part of each kick decay like the impact part?"""
import numpy as np, pandas as pd
from scipy.optimize import curve_fit
from common import *
n='round3'
o=L(n,'orders'); s=pd.read_parquet('cache/round3_s_at_insert.parquet'); o['s']=s.s.to_numpy(); v=regime_1s(n)
g=L(n,'strue'); x=g.sp.to_numpy()
def at(t): return x[np.clip((np.asarray(t)*10).astype(int),0,len(x)-1)]
oi=o.reset_index(drop=True)
mk=~oi.ioc.to_numpy(); S=oi.s.to_numpy(); t=oi.t.to_numpy()
# previous and next maker insert S around each taker
idx_mk=np.where(mk)[0]
k=oi[oi.ioc&(oi.filled>0)]
ki=k.index.to_numpy()
p=idx_mk[np.searchsorted(idx_mk,ki)-1]; q=idx_mk[np.minimum(np.searchsorted(idx_mk,ki),len(idx_mk)-1)]
kick=S[q]-S[p]
sg=np.where(k.side==0,1,-1); f=k.filled.to_numpy()**0.62; vr=v[k.t.astype(int)]
bad=np.zeros(len(k),bool)
for on in shocks(n).onset: bad|=(k.t.to_numpy()>on-62)&(k.t.to_numpy()<on+8)
bad|=(t[q]-t[p])>0.05
for reg in [False,True]:
    sel=(vr==reg)&~bad
    c=np.sum(kick[sel]*sg[sel]*f[sel])/np.sum(f[sel]**2)
    pred=c*sg[sel]*f[sel]; noise=kick[sel]-pred
    print(f'vol={reg} n={sel.sum()} c_immediate={c:.4f} noise sd={noise.std():.3f}')
    t0=t[q][sel]
    for h in [5,10,20,30,60]:
        fut=at(t0+h)-S[q][sel]
        # regress future change on pred and noise jointly
        A=np.c_[pred,noise]; b,*_=np.linalg.lstsq(A,fut,rcond=None)
        print(f'   +{h}s: future move / pred-impact = {b[0]:+.3f}, / noise = {b[1]:+.3f}')
