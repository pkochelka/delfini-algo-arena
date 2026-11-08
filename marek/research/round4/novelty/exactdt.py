"""Scan for over-represented exact microsecond gaps between event classes (r2 vs r3)."""
import numpy as np, pandas as pd
from common import *
def scan(a,b,maxdt=20000,lags=12,same=False):
    # count exact dt = b_j - a_i for b after a within maxdt
    a=np.sort(a); b=np.sort(b); cnt={}
    idx=np.searchsorted(b,a,side='right')
    for l in range(lags):
        j=idx+l; ok=j<len(b); d=b[j[ok]]-a[ok]; d=d[(d>0)&(d<=maxdt)]
        v,c=np.unique(d,return_counts=True)
        for x,y in zip(v,c): cnt[x]=cnt.get(x,0)+y
    s=pd.Series(cnt); bg=s.median()
    return s.sort_values(ascending=False).head(6), bg
for n in ['round3','round4']:
    o=L(n,'orders'); c=L(n,'cancels')
    tus=lambda x: np.rint(x*1e6).astype(np.int64)
    mk=tus(o.t[~o.ioc].to_numpy()); tk=tus(o.t[o.ioc].to_numpy()); cc=tus(c.t.to_numpy())
    for nm,(a,b,l) in {'tk->tk':(tk,tk,3),'mk->tk':(mk,tk,3),'tk->mk':(tk,mk,8),'tk->cancel':(tk,cc,8),'cancel->cancel':(cc,cc,8),'cancel->mk':(cc,mk,8)}.items():
        s,bg=scan(a,b,lags=l)
        print(n,nm,'bg',bg,' top:',s.to_dict())
