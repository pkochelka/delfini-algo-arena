"""Find 2ms-spaced chains of maker inserts (the 'layering' participant)."""
import numpy as np, pandas as pd, sys
for n in ['round2','round3']:
    o=pd.read_parquet(f'cache/{n}_orders.parquet')
    tus=np.rint(o.t.to_numpy()*1e6).astype(np.int64)
    o['tus']=tus
    m=o[~o.ioc]
    # exact-dt histogram between any pair of maker inserts within 10ms, by dt in us (check which dts are overrepresented)
    t=m.tus.to_numpy(); 
    cnt={}
    for lag in range(1,8):
        d=t[lag:]-t[:-lag]
        v,c=np.unique(d[d<=10000],return_counts=True)
        for a,b in zip(v,c): cnt[a]=cnt.get(a,0)+b
    s=pd.Series(cnt).sort_values(ascending=False)
    print(n,'most common exact dt (us) between maker inserts within 10ms:'); print(s.head(12).to_string())
