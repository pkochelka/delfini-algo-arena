"""Our passive fills (r3_dev_best full log) by price relative to the filling taker's own limit L, per regime."""
import pandas as pd, numpy as np, sys
tr=pd.read_csv(sys.argv[1]+'/trades.csv')
tk=pd.read_parquet('marek/research/round3/ideas/ceiling/_cache/ceil_takers.parquet')
s=pd.read_parquet('marek/research/round3/s_dynamics/s_path.parquet')
vs=pd.read_csv('marek/research/round3/s_dynamics/vol_segments.csv'); sh=pd.read_csv('marek/research/round3/flow/shock_events.csv')
stt=s.t_us.values; sv=s.s_hat.values
S_at=lambda us: sv[np.clip(np.searchsorted(stt,us,side='right')-1,0,len(sv)-1)]
my=tr[(tr.maker_owner=='user')&(tr.taker_owner=='market')].copy()
tk['taker_order_id']=tk.ins_i  # check mapping below
# map taker by timestamp (unique per taker event)
tk2=tk.set_index('t_us')
my=my[my.timestamp_us.isin(tk2.index)]
j=tk2.loc[my.timestamp_us]
my['lim']=j.lim.values; my['S_after']=j.S_after.values; my['tq']=j.q.values; my['tfilled']=j.filled.values
sg=np.where(my.taker_side=='B',1,-1)   # taker buys -> we sold
my['k']=sg*(my.lim-my.price)  # 0 = at the taker limit, 1 = one tick inside
x=my.timestamp_us.values/1e6
vol=np.zeros(len(x),bool)
for a,b in zip(vs.start,vs.end): vol|=(x>=a)&(x<b)
shk=np.zeros(len(x),bool)
for o in sh.onset: shk|=(x>=o-1)&(x<o+20)
my['reg']=np.where(shk,'shock',np.where(vol,'vol','calm'))
for h in [0,1,5]:
    my[f'm{h}']=sg*(my.price-(my.S_after if h==0 else S_at(my.timestamp_us.values+h*1_000_000)))
my['resid']=my.tq>my.tfilled
for (r,k),d in my.groupby(['reg','k']):
    if len(d)<20: continue
    w=d.quantity
    print(f'{r:6s} k={k:3d} fills={len(d):6d} lots={w.sum():7d} resid_share={np.average(d.resid,weights=w):.2f} '+' '.join(f'm{h}={np.average(d[f"m{h}"],weights=w):+.2f}' for h in [0,1,5]))
print('unmatched trades', (tr.maker_owner=='user').sum()-len(my))
