import numpy as np, pandas as pd
B=5
for n in ['round2','round3']:
    o=pd.read_parquet(f'cache/{n}_orders.parquet'); m=o[~o.ioc].copy()
    mid=(m.pbid+m.pask)/2
    m['off']=np.where(m.side==0, mid-m.price, m.price-mid); m['life']=m.end_t-m.t
    nb=6000//B; deep=np.bincount((m.t[m.off>=3.5]//B).astype(int),minlength=nb)[:nb]; on=deep>=150
    m['on']=on[(m.t//B).astype(int).clip(0,nb-1)]
    c=m[~m.on]
    print(n,'calm: log-vol quantiles',np.percentile(np.log(c.vol),[1,5,10,25,50,75,90,95,99]).round(3))
    print(' vol==1 frac',(c.vol==1).mean(),' vol<=3',(c.vol<=3).mean(), 'mean vol', c.vol.mean())
    for lo,hi in [(0,2),(2,5),(5,10),(10,1e9)]:
        s=c[(c.life>=lo)&(c.life<hi)]
        print(f'  life[{lo},{hi}) n={len(s)} frac={len(s)/len(c):.4f} lv={np.log(s.vol).mean():.3f} off_mean={s.off.mean():.2f} filled={(s.filled>0).mean():.3f} offq={s.off.quantile([.1,.5,.9]).round(2).tolist()}')
