"""Does filled volume against ladder orders move S like regular fills? Immediate kick regression (exact S)."""
import numpy as np, pandas as pd
from common import *
n='round3'
o=L(n,'orders').reset_index(drop=True); S=pd.read_parquet('cache/round3_s_at_insert.parquet').s.to_numpy(); t=o.t.to_numpy(); v=regime_1s(n)
ch=pd.read_parquet('cache/round3_makers_chains.parquet')[['row','chain']]
tr=L(n,'trades'); tr=tr.merge(ch.rename(columns={'row':'maker_row'}),on='maker_row',how='left'); tr['chain']=tr.chain.fillna(False)
lad=tr[tr.chain].groupby('taker_row').qty.sum(); tot=tr.groupby('taker_row').qty.sum()
row2idx=pd.Series(o.index.to_numpy(),index=o.row.to_numpy())
k=o[o.ioc&(o.filled>0)].copy(); k['lad']=k.row.map(lad).fillna(0); k['tot']=k.row.map(tot).fillna(0)
mk=~o.ioc.to_numpy(); idx_mk=np.where(mk)[0]; ki=k.index.to_numpy()
p=idx_mk[np.searchsorted(idx_mk,ki)-1]; q=idx_mk[np.minimum(np.searchsorted(idx_mk,ki),len(idx_mk)-1)]
kick=S[q]-S[p]; sg=np.where(k.side==0,1,-1)
ok=((t[q]-t[p])<0.05)
bad=np.zeros(len(k),bool)
for on in shocks(n).onset: bad|=(k.t.to_numpy()>on-1)&(k.t.to_numpy()<on+8)
ok&=~bad
vr=v[k.t.astype(int)]
print('takers with ladder fills:',(k.lad>0).mean().round(3),' ladder share of filled volume:',(k.lad.sum()/k.tot.sum()).round(3))
for r in [False,True]:
    s=ok&(vr==r)&(k.lad>0)
    tot_=k.tot.to_numpy()[s]; reg=(k.tot-k.lad).to_numpy()[s]; y=kick[s]*sg[s]
    for nm,xx in [('total^.62',tot_**0.62),('regular-only^.62',reg**0.62)]:
        b=np.sum(xx*y)/np.sum(xx*xx); r2=1-np.sum((y-b*xx)**2)/np.sum((y-y.mean())**2)
        print(f' vol={r} n={s.sum()} model {nm}: c={b:.4f} R2={r2:.3f}')
