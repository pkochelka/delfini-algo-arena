import numpy as np, pandas as pd
m=pd.read_parquet('cache/round3_makers_chains.parquet')
top=np.load('cache/round3_top.npz'); tt=top['t']; bid=top['bid'].astype(float); ask=top['ask'].astype(float)
bid[bid<=0]=np.nan; ask[ask<=0]=np.nan; mid=(bid+ask)/2
m['touchoff']=np.where(m.side==0, m.pbid-m.price, m.price-m.pask)
on=np.load('cache/round3_on5s.npy'); m['on']=on[(m.t//5).astype(int).clip(0,1199)]
C=m[m.chain].sort_values(['cid','cidx'])
g=C.groupby('cid').agg(t=('t','first'),row=('row','first'),side=('side','first'),start=('touchoff','first'),n=('vol','size'),on=('on','first'))
def midat(t): 
    i=np.searchsorted(tt,t,side='right')-1; return mid[np.clip(i,0,len(mid)-1)]
for lag in [0.05,0.2,1.0]:
    g[f'dm{lag}']=(mid[g.row.to_numpy()-1]-midat(g.t.to_numpy()-lag))*np.where(g.side==0,1,-1)  # + = price moved away from this side (bids: up)
g['fwd1']=(midat(g.t.to_numpy()+1.0)-mid[g.row.to_numpy()-1])*np.where(g.side==0,1,-1)
g['typ']=np.where(g.start==2,'start2','refill_out')
print(g.groupby(['on','typ'])[['dm0.05','dm0.2','dm1.0','fwd1']].mean().round(3))
print(g.groupby(['on','typ']).size())
# random baseline: same for random times
