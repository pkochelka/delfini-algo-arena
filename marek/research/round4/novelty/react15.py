import numpy as np, pandas as pd
from common import *
o=L('round4','orders'); o=o.assign(tus=np.rint(o.t*1e6).astype(np.int64))
k=L('round4','takers_tw'); k=k.assign(tus=np.rint(k.t*1e6).astype(np.int64))
m=o[~o.ioc]; pos=dict(zip(k.tus,range(len(k))))
j=np.array([pos.get(t-1,-1) for t in m.tus]); hit=j>=0
print('makers 1us after taker',hit.sum())
kk=k.iloc[j[hit]]; mm=m[hit]
print('trigger is TWAP child',kk.tw.mean().round(3),' n',kk.tw.sum(),' of TWAP children',k.tw.sum(),' frac TWAP children triggering',(kk.tw.sum()/k.tw.sum()).round(3))
print('vol of reactive maker when trigger TWAP:',mm.vol[kk.tw.to_numpy()].value_counts().head(5).to_dict(),' non-TWAP:',mm.vol[~kk.tw.to_numpy()].value_counts().head(5).to_dict())
print('trigger taker sizes (non TWAP) quantiles',np.percentile(kk.vol[~kk.tw.to_numpy()],[10,50,90]))
# which TWAP episodes trigger
e=kk[kk.tw].groupby('tw_cid').size(); print('episodes with triggers',e.to_dict())
# non-twap triggers: time clusters
nt=kk[~kk.tw.to_numpy()]; print('non-TWAP trigger times (first 40)',nt.t.round(1).tolist()[:40])
