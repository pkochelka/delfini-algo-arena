import numpy as np, pandas as pd
from common import *
for n in ['round2','round3']:
    o=L(n,'orders'); m=o[~o.ioc].reset_index(drop=True)
    f=m[(m.end==2)]  # filled fully
    # next maker insert same side same price within W after fill end
    key=m.side.astype(np.int64)*100000+m.price
    out=[]
    for W in [0.01,0.05,0.2]:
        hits=0; same=0; tot=0
        ts=m.t.to_numpy(); ks=key.to_numpy(); vs=m.vol.to_numpy()
        order=np.lexsort((ts,ks)); ks_s=ks[order]; ts_s=ts[order]; vs_s=vs[order]
        fk=(f.side.astype(np.int64)*100000+f.price).to_numpy(); fe=f.end_t.to_numpy(); fv=f.vol.to_numpy()
        lo=np.searchsorted(ks_s*1e5+ts_s, fk*1e5+fe, side='right')
        # careful approach: combined key
        comb=ks_s.astype(np.float64)*1e5+ts_s
        i=np.searchsorted(comb,fk*1e5+fe,side='right')
        ok=(i<len(comb)); ok[ok]&=(ks_s[i[ok]]==fk[ok])&(ts_s[i[ok]]-fe[ok]<=W)
        hits=ok.mean(); same=(ok&(np.where(ok,vs_s[np.minimum(i,len(vs_s)-1)],-1)==fv)).mean()
        # baseline: random times shifted by +1s
        j=np.searchsorted(comb,fk*1e5+fe+1.0,side='right'); okb=(j<len(comb)); okb[okb]&=(ks_s[j[okb]]==fk[okb])&(ts_s[j[okb]]-fe[okb]-1.0<=W)
        out.append(f'W={W}: reinsert={hits:.4f} (baseline {okb.mean():.4f}) same-size={same:.5f}')
    print(n,len(f),' | '.join(out))
