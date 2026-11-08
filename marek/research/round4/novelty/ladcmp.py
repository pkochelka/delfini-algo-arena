import numpy as np, pandas as pd
from common import *
for n in ['round3','round4']:
    m=pd.read_parquet(f'cache/{n}_makers_chains.parquet'); v=regime_1s(n)
    c=m[m.chain]; c=c.assign(vr=v[c.t.astype(int)])
    b=c.groupby('cid').agg(lv=('vol',lambda x:np.log(x).mean()),vr=('vr','first'),n=('vol','size'),t=('t','first'))
    for r in [False,True]:
        x=b[b.vr==r]; T=(v[:6000]==r).sum()
        print(n,'vol' if r else 'calm',f'bursts/s={len(x)/T:.2f} logB mean={x.lv.mean():.3f} sd={x.lv.std():.3f} median B={np.exp(x.lv.median()):.0f} len mean={x.n.mean():.2f} len>=7 {(x.n>=7).sum()}')
    c=c.assign(off=np.where(c.side==0,c.pbid-c.price,c.price-c.pask))
    print('   rung offset from same-side touch dist',c.off.value_counts().sort_index().loc[-2:12].to_dict())
