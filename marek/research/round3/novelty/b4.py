import numpy as np, pandas as pd
o=pd.read_parquet('cache/round3_orders.parquet'); m=o[~o.ioc].copy()
mid=(m.pbid+m.pask)/2
m['off']=np.where(m.side==0, mid-m.price, m.price-mid); m['life']=m.end_t-m.t
m['b']=(m.t//5).astype(int)
s=m.groupby('b').agg(n=('vol','size'),deep=('off',lambda x:(x>=3.5).sum()))
x=s.deep.to_numpy()
# print as compact string of 5s bins, scaled
line=''.join(' .:-=+*#%@'[min(9,int(v/25))] for v in x)
for i in range(0,len(line),120): print(f'{i*5:5d}s',line[i:i+120])
print(np.percentile(x,[5,25,50,75,95,99]))
