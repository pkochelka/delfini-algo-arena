import numpy as np, pandas as pd
from common import *
for n in ['round3','round4']:
    o=L(n,'orders'); v=regime_1s(n); s=pd.read_parquet(f'cache/{n}_s_at_insert.parquet'); o['s']=s.s.to_numpy()
    m=o[~o.ioc].copy()
    if True:
        ch=pd.read_parquet(f'cache/{n}_makers_chains.parquet')[['oid','chain']]; m=m.merge(ch,on='oid',how='left'); m=m[~m.chain]
    m['life']=m.end_t-m.t; m['D']=np.where(m.side==0,m.s-m.price,m.price-m.s)
    vr=v[m.t.astype(int).clip(0,5999)]
    for r in [False,True]:
        x=m[(vr==r)&(m.end==1)]
        y=m[vr==r]
        print(f'{n} vol={r} regular makers: life q25/50/75/95/99={np.percentile(x.life,[25,50,75,95,99]).round(3).tolist()} D q10/50/90={np.percentile(y.D.dropna(),[10,50,90]).round(2).tolist()} lv={np.log(y.vol).mean():.3f} fill={(y.filled>0).mean():.3f} vol>=90 frac={(y.vol>=90).mean():.4f}')
