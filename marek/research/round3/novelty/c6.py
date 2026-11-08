import numpy as np, pandas as pd
m=pd.read_parquet('cache/round3_makers_chains.parquet'); m['life']=m.end_t-m.t
c=pd.read_parquet('cache/round3_cancels.parquet'); top=np.load('cache/round3_top.npz')
bid=top['bid'].astype(float); ask=top['ask'].astype(float)
crow=dict(zip(c.oid.to_numpy(), c.row.to_numpy()))
m['crow']=m.oid.map(crow)
on=np.load('cache/round3_on5s.npy'); m['on']=on[(m.t//5).astype(int).clip(0,1199)]
m['touchoff']=np.where(m.side==0, m.pbid-m.price, m.price-m.pask)
x=m[(m.end==1)&m.crow.notna()].copy()
r=x.crow.astype(int).to_numpy()-1
x['cbid']=bid[r]; x['cask']=ask[r]
x['touchoff_c']=np.where(x.side==0, x.cbid-x.price, x.price-x.cask)
x['oppoff_c']=np.where(x.side==0, x.cask-x.price, x.price-x.cbid)
for flag in [False,True]:
  for ch in [True,False]:
    s=x[(x.chain==ch)&(x.on==flag)]
    d=(s.touchoff_c-s.touchoff)
    print(f'on={flag} chain={ch} n={len(s)} change in touch-dist at cancel (neg=price approached): q={np.percentile(d,[5,25,50,75,95]).tolist()} mean={d.mean():.3f}  frac dist_c>7: {(s.touchoff_c>7).mean():.3f} frac dist_c<2:{(s.touchoff_c<2).mean():.3f}')
# hazard by current touch distance for chain orders: crude - cancel rate vs touchoff_c
s=x[x.chain]
print('chain cancels: touchoff_c dist',s.touchoff_c.clip(-2,12).value_counts(normalize=True).sort_index().round(3).to_dict())
