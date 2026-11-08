import numpy as np, pandas as pd
from common import *
o=L('round4','orders'); k=o[o.ioc].copy(); v=regime_1s('round4')
k['sg']=np.where(k.side==0,1,-1)
a=k[k.vol==160]; print('n160',len(a),'side buy frac',(a.side==0).mean())
print('time quantiles',np.percentile(a.t,[0,10,25,50,75,90,100]).round(1))
ia=np.diff(a.t.to_numpy()); print('interarrival q',np.percentile(ia,[1,5,10,25,50,75,90,99]).round(3), 'cv',ia.std()/ia.mean())
# runs of sides
s=a.sg.to_numpy(); ch=np.r_[True,s[1:]!=s[:-1]]; rl=np.diff(np.r_[np.where(ch)[0],len(s)]); print('run lengths of side among 160s: mean',rl.mean(),'dist',np.bincount(rl)[:15].tolist())
print('sign acf lag1..3', [round(np.corrcoef(s[l:],s[:-l])[0,1],3) for l in (1,2,3)])
print('vol regime frac',v[a.t.astype(int)].mean(),' overall',v[:6000].mean())
# limit vs touch
tch=np.where(a.side==0,a.pask,a.pbid); lt=a.sg*(a.price-tch); print('limit - touch dist',lt.value_counts().sort_index().head(15).to_dict())
print('filled frac',(a.filled>0).mean(),'full',(a.filled==a.vol).mean())
# per-minute counts
c=np.bincount((a.t//60).astype(int),minlength=100); print('per minute counts',c.tolist())
# within 1us of other events?
print('time frac part us', (np.rint(a.t*1e6)%1000).describe())
