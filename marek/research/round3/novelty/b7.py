import numpy as np, pandas as pd
o=pd.read_parquet('cache/round3_orders.parquet'); g=pd.read_parquet('cache/round3_grid.parquet')
m=o[~o.ioc].copy(); mid=(m.pbid+m.pask)/2
m['off']=np.where(m.side==0, mid-m.price, m.price-mid); m['life']=m.end_t-m.t
L=m[(m.life>3)&(m.off>=2.5)].copy()
print('n',len(L))
print(L.head(40)[['t','oid','side','price','vol','pbid','pask','off','life','filled','end']].to_string())
# inter-arrival among L
ia=np.diff(L.t.to_numpy()); print('ia quantiles',np.percentile(ia,[1,5,25,50,75,95]).round(4))
# mid at end time vs at insert
gm=g.mid.ffill().to_numpy()
endmid=gm[np.clip((L.end_t*10).astype(int),0,59999)]
L['off_end']=np.where(L.side==0, endmid-L.price, L.price-endmid)
print('off at insert vs end',L.off.describe().round(2).to_dict(), L.off_end.describe().round(2).to_dict())
print('vol hist', np.percentile(L.vol,[5,25,50,75,95,99]))
