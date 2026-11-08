import numpy as np, pandas as pd
m=pd.read_parquet('cache/round3_makers_chains.parquet')
mid=(m.pbid+m.pask)/2
m['off']=np.where(m.side==0, mid-m.price, m.price-mid); m['life']=m.end_t-m.t
m['touchoff']=np.where(m.side==0, m.pbid-m.price, m.price-m.pask)  # ticks behind same-side touch
C=m[m.chain]; N=m[~m.chain]
print('chain orders:',len(C),' frac of makers',len(C)/len(m))
print('chain vol quantiles',np.percentile(C.vol,[0,1,5,25,50,75,95,99,100]))
print('non-chain vol quantiles',np.percentile(N.vol,[1,5,25,50,75,95,99]))
print('chain life quantiles',np.percentile(C.life.dropna(),[1,5,25,50,75,95,99]).round(3))
print('chain touchoff dist', C.touchoff.value_counts().sort_index().head(15).to_dict())
print('chain off (from mid) dist', C.off.value_counts().sort_index().head(20).to_dict())
# price steps within chain
C=C.sort_values(['cid','cidx'])
dp=C.groupby('cid').price.diff().dropna()
sd=np.where(C.side==0,-1,1)[C.groupby('cid').cumcount()>0]
print('price step (outward +):',pd.Series(dp.to_numpy()*sd).value_counts().head(10).to_dict())
# first order offset per chain
f=C[C.cidx==0]
print('first order touchoff',f.touchoff.value_counts().sort_index().head(12).to_dict())
print('first order off from mid',f.off.value_counts().sort_index().head(12).to_dict())
print('end reasons', C.end.value_counts().to_dict())
