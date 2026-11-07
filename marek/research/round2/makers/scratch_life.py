import numpy as np, pandas as pd
from prep import load
o=load(); g=o[~o.ioc].copy()
life=(g.cancel_us-g.t_us)/1e6
print('cancel row present', g.cancel_us.notna().mean())
print('life quantiles', life.quantile([.01,.1,.25,.5,.75,.9,.99]).round(4).to_dict(), 'mean', life.mean())
fl=g.end=='filled'
print('filled share of makers', fl.mean(), 'filled & cancel row', (fl & g.cancel_us.notna()).sum()/g.cancel_us.notna().sum())
print('partial filled then cancelled', ((g.filled>0)&(g.end=='cancelled')).mean())
# lifetime vs size
g['lv']=np.log10(g.volume)
print(life.groupby(pd.cut(g.volume,[0,10,100,1000,5000,20000,1e6])).agg(['mean','median','count']))
# by side
print(life.groupby(g.sg).mean())
# by placement relative to touch at insert: bid: price - bid
rel=np.where(g.sg==1, g.price-g.bid, g.ask-g.price)
print(life.groupby(pd.Series(rel,index=g.index).clip(-5,2)).agg(['mean','median','count']))
# by time segment
print(life.groupby((g.t_us//600e6)).mean().round(3).to_dict())
# hypoexp fit of life
from scipy import optimize
L=life.dropna().to_numpy(); L=L[L>0]
def nll(th):
    a,b=np.exp(th)
    d=a*b/(b-a)*(np.exp(-a*L)-np.exp(-b*L)); return -np.log(np.maximum(d,1e-300)).sum()
r=optimize.minimize(nll,np.log([2.5,30]),method='Nelder-Mead'); print('hypoexp',np.exp(r.x))
# min lifetime
print('min life', L.min(), (L<0.001).mean(), (L<0.005).mean())
