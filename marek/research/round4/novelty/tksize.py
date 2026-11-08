import numpy as np, pandas as pd
from scipy import stats
from common import *
for n in ['round3','round4']:
    o=L(n,'orders'); k=o[o.ioc]
    vc=k.vol.value_counts()
    # expected count under lognormal fitted
    mu,sd=np.log(k.vol).mean(),np.log(k.vol).std()
    def exp_cnt(v): return len(k)*(stats.norm.cdf((np.log(v+0.5)-mu)/sd)-stats.norm.cdf((np.log(np.maximum(v-0.5,1e-9))-mu)/sd))
    vals=np.arange(1,3000); e=exp_cnt(vals); obs=vc.reindex(vals).fillna(0).to_numpy()
    z=(obs-e)/np.sqrt(e+1)
    top=np.argsort(z)[::-1][:20]
    print(n,'n',len(k),'mu,sd',round(mu,3),round(sd,3))
    print('  top excess sizes (size obs exp):',[(int(vals[i]),int(obs[i]),round(e[i],1)) for i in top])
    for M in [5,10,25,50,100]:
        sel=(k.vol>=50)&(k.vol<=2000); print(f'  frac mult {M} among 50..2000: {(k.vol[sel]%M==0).mean():.4f} (uniform {1/M:.4f})')
    print('  >5000',(k.vol>5000).sum(),'exp',round(len(k)*(1-stats.norm.cdf((np.log(5000)-mu)/sd)),1),' >2000',(k.vol>2000).sum(),'exp',round(len(k)*(1-stats.norm.cdf((np.log(2000)-mu)/sd)),1))
