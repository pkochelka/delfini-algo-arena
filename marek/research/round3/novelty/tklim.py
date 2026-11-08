import numpy as np, pandas as pd
from common import *
for n in ['round2','round3']:
    o=L(n,'orders'); s=pd.read_parquet(f'cache/{n}_s_at_insert.parquet'); o['s']=s.s.to_numpy(); v=regime_1s(n)
    k=o[o.ioc].copy(); sg=np.where(k.side==0,1,-1); k['g']=sg*(k.price-k.s)
    bad=np.zeros(len(k),bool)
    for a in shocks(n).onset: bad|=(k.t>a-0.5)&(k.t<a+8)
    k=k[~bad]; vr=v[k.t.astype(int)]
    for r in [False,True]:
        g=k.g[vr==r]
        band=1 if not r else 2.55
        reg=(g<=band+0.5)
        print(f'{n} vol={r} n={len(g)} frac<=band+.5: {reg.mean():.3f}  tail excess (g-band-0.5) mean={((g-band-0.5)[~reg]).mean():.2f} q90/99 of g: {np.percentile(g,[90,95,99,99.9]).round(2).tolist()} below S: {(g<0).mean():.4f}')
