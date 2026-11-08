import numpy as np, pandas as pd
for n in ['round2','round3']:
    o=pd.read_parquet(f'cache/{n}_orders.parquet'); m=o[~o.ioc].copy()
    m['life']=m.end_t-m.t
    m['dcl']=m.dist.clip(-8,1).round()
    print(n)
    print(m.groupby('dcl').agg(n=('vol','size'),frac=('vol',lambda x:len(x)/len(m)),lv=('vol',lambda x:np.log(x).mean()),life_med=('life','median'),life_mean=('life','mean'),p99=('life',lambda x:x.quantile(.99)),fill=('filled',lambda x:(x>0).mean())).to_string(float_format=lambda x:f'{x:.3f}'))
    # life histogram tail
    l=m.life[m.end==1]
    print('life >2s',(l>2).mean(),'>5s',(l>5).mean(),'>10s',(l>10).mean(),'>30',(l>30).mean(), 'max',l.max())
