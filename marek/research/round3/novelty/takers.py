import numpy as np, pandas as pd
from common import *
for n in ['round2','round3']:
    o=L(n,'orders'); k=o[o.ioc].copy(); v=regime_1s(n); sh=shocks(n)
    k['vol_r']=v[k.t.astype(int)]
    ins=np.zeros(len(k),bool)
    for a in sh.onset: ins|=(k.t>=a)&(k.t<a+8)
    k['shk']=ins
    sg=np.where(k.side==0,1,-1)
    k['lim_touch']=sg*(k.price-np.where(k.side==0,k.pask,k.pbid))
    k['lim_mid']=sg*(k.price-(k.pask+k.pbid)/2)
    print(n)
    for r in [False,True]:
        s=k[(k.vol_r==r)&~k.shk]; T=((v[:6000]==r).sum())
        print(f'  vol={r} rate={len(s)/T:.2f} lv={np.log(s.vol).mean():.3f} sd={np.log(s.vol).std():.3f} fill_any={(s.filled>0).mean():.3f} lim-touch q={s.lim_touch.quantile([.05,.25,.5,.75,.95]).tolist()} lim-mid mean={s.lim_mid.mean():.2f} part={((s.filled>0)&(s.filled<s.vol)).mean():.3f}')
        print('     lim-mid hist', s.lim_mid.clip(-1,8).round(1).value_counts(normalize=True).sort_index().round(3).head(18).to_dict())
    print('  sizes >5000:',(k.vol>5000).sum(),' >20000:',(k.vol>20000).sum(), ' expected lognormal(4.10,1.40) >5000:', len(k)*(1-__import__('scipy').stats.norm.cdf((np.log(5000)-4.10)/1.40)))
