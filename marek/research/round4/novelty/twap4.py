import numpy as np, pandas as pd
from scipy import stats
from common import *
e=pd.read_csv('twap_episodes.csv'); k=L('round4','takers_tw'); x=L('round4','strue').sp.to_numpy()
def S(t): return np.interp(t,np.arange(len(x))*0.1,x)
n=np.sort(e.n.to_numpy()); print('n sorted',n.tolist()); print('KS vs U(22,58):',stats.kstest(n,stats.randint(22,59).cdf).pvalue.round(3), ' total qty n*160 quantiles',np.percentile(n*160,[0,50,100]))
print('dur*2+1 == n ?', np.all(np.isclose(e.dur*2+1,e.n)))
tw=k[k.tw]; sg=np.where(tw.side==0,1,-1); t=tw.t.to_numpy()
for j0,j1 in [(2,5),(5,10),(10,20),(20,30),(30,60)]:
    m=(tw.tw_idx>=j0)&(tw.tw_idx<j1)
    out=[np.mean(sg[m]*(S(t[m]+h)-S(t[m]))) for h in [1,2,5,10]]
    print(f'child idx {j0}-{j1}: signed S move next 1/2/5/10 s =',np.round(out,2).tolist(),' n',m.sum())
# after end
t1=e.t1.to_numpy()+0.5; s1=np.where(e.side=='B',1,-1)
print('after last child: signed S move +2/5/10/20/30 s:',[round(np.mean(s1*(S(t1+h)-S(t1))),2) for h in [2,5,10,20,30]])
# fill of children: filled qty & price relative to S
print('TWAP limit - S (signed) quantiles', np.percentile(sg*(tw.price.to_numpy()-S(t)),[5,25,50,75,95]).round(2))
