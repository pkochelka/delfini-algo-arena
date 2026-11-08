"""Crude S proxy on a 0.1 s grid: midpoint of max regular bid insert and min regular ask insert per 0.2 s bin (no chains)."""
import numpy as np, pandas as pd
from common import *
W=0.2
for n in ['round4']:
    o=L(n,'orders'); m=o[~o.ioc]
    ch=pd.read_parquet(f'cache/{n}_makers_chains.parquet')[['oid','chain']]; m=m.merge(ch,on='oid',how='left'); m=m[~m.chain.fillna(False)]
    m=m[~((m.vol==15))]  # drop reactive 15-lot maker
    grid=np.arange(0,6000,0.1); out={}
    for sd,f in [(0,'max'),(1,'min')]:
        s=m[m.side==sd]; b=np.floor(s.t/W).astype(int); agg=s.groupby(b).price.agg(f)
        nb=int(6000/W)+1; arr=np.full(nb,np.nan); arr[agg.index.to_numpy()]=agg.to_numpy(); out[sd]=arr
    px=(out[0]+out[1])/2; gi=(grid/W).astype(int)
    pd.DataFrame(dict(t=grid,sp=pd.Series(px[gi]).ffill().bfill().to_numpy())).to_parquet(f'cache/{n}_sproxy.parquet'); print(n,'done')
