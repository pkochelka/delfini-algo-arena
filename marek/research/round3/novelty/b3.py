import numpy as np, pandas as pd
for n in ['round2','round3']:
    o=pd.read_parquet(f'cache/{n}_orders.parquet'); m=o[~o.ioc].copy()
    m['life']=m.end_t-m.t
    mid=(m.pbid+m.pask)/2
    m['off']=np.where(m.side==0, mid-m.price, m.price-mid)  # distance from mid, positive = away
    print(n, 'offset from mid hist (0.5 bins)')
    h=pd.cut(m.off,np.arange(-1,12.5,0.5)).value_counts().sort_index()
    print((h/len(m)).round(4).to_string())
    m['min']=(m.t//60).astype(int)
    deep=(m.off>=2.5)
    pm=m.groupby('min').agg(n=('vol','size'),deep=('off',lambda x:(x>=2.5).sum()),lv=('vol',lambda x:np.log(x).mean()))
    print(pm.T.to_string() if n=='round3' else pm.describe().to_string())
