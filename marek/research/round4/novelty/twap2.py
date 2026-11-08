import numpy as np, pandas as pd
from common import *
k=L('round4','takers_tw'); g=pd.read_csv('twap_episodes.csv')
s=np.where(k.side==0,1,-1)
def acf(x,l): return np.corrcoef(x[l:],x[:-l])[0,1]
print('sign acf all',[round(acf(s,l),3) for l in (1,2,3,5,10)])
r=s[~k.tw.to_numpy()]; print('sign acf non-TWAP',[round(acf(r,l),3) for l in (1,2,3,5,10)])
# non-twap takers inside twap windows: side bias?
inw=np.zeros(len(k),bool); ws=np.zeros(len(k))
for _,e in g.iterrows():
    m=(k.t>=e.t0)&(k.t<=e.t1+0.5); inw|=m; ws[m]=1 if e.side=='B' else -1
nt=~k.tw.to_numpy()
print('non-TWAP takers inside windows',(inw&nt).sum(),' mean sign*episode side',np.mean(s[inw&nt]*ws[inw&nt]).round(4))
# exact-dt scan among all takers (excluding TWAP): any other periodic dt?
tus=np.rint(k.t*1e6).astype(np.int64).to_numpy()[nt]
cnt={}
for lag in range(1,40):
    d=tus[lag:]-tus[:-lag]; d=d[d<=3_000_000]
    v,c=np.unique(d,return_counts=True)
    for a,b in zip(v,c):
        if b>=3: cnt[a]=cnt.get(a,0)+b
print('non-TWAP taker exact dt with count>=3:',sorted(cnt.items(),key=lambda x:-x[1])[:15])
# all takers on exact multiples of 500000 from any other taker with same size
tusa=np.rint(k.t*1e6).astype(np.int64).to_numpy(); vol=k.vol.to_numpy()
pos={t:i for i,t in enumerate(tusa)}
hits=[]
for i in range(len(k)):
    j=pos.get(tusa[i]-500000)
    if j is not None: hits.append((vol[j],vol[i]))
h=pd.DataFrame(hits,columns=['a','b']); print('pairs 500000us apart:',len(h),' same size',(h.a==h.b).sum(),' sizes',h[h.a==h.b].a.value_counts().head().to_dict())
