import pandas as pd, numpy as np
SP='marek/research/round2/liquidity/cache'  # holds cf_logs/ from a full-log run of the cf config
tr=pd.read_csv(SP+'/cf_logs/trades.csv')
me=tr[(tr.taker_owner!='market')|(tr.maker_owner!='market')].copy()
me['taker']=me.taker_owner!='market'
# our side: if we're taker, our side = taker_side (B buy); else opposite
ts=np.where(me.taker_side=='B',1,-1)
me['sgn']=np.where(me.taker,ts,-ts)
S=pd.read_parquet('marek/research/round2/s_dynamics/s_path.parquet')
st=S.t_us.values; sh=S.s_hat.values; vol=S.vol.values; shk=S.shock.values
def s_at(t):
    i=np.searchsorted(st,t,side='right')-1; return sh[np.clip(i,0,len(sh)-1)]
t=me.timestamp_us.values
i=np.clip(np.searchsorted(st,t,side='left')-1,0,len(sh)-1)  # S before the fill's event
me['S0']=sh[i]; me['vol']=vol[i]
for h in [0.1,1,5,30]:
    me[f'S{h}']=s_at(t+int(h*1e6))
me['edge0']=me.sgn*(me.S0-me.price)*me.quantity
for h in [0.1,1,5,30]: me[f'mk{h}']=me.sgn*(me[f'S{h}']-me.price)*me.quantity
# shock windows
sk=pd.read_csv('marek/research/round2/flow/shock_events.csv').onset.values*1e6
near=np.zeros(len(me),bool)
for o in sk: near|=(t>=o-1e6)&(t<o+20e6)
me['reg']=np.where(near,'shock',np.where(me.vol,'vol','calm'))
g=me.groupby(['reg','taker'])
print(g[['quantity','edge0','mk0.1','mk1','mk5','mk30']].sum().round(0))
print(g.size())
# inventory
pos=np.cumsum(me.sgn*me.quantity).values
me['pos']=pos
# time-weighted |pos| by regime on 0.1s grid
grid=np.arange(0,6000e6,1e5)
j=np.searchsorted(t,grid,side='right')-1
pg=np.where(j>=0,pos[np.clip(j,0,None)],0)
gi=np.clip(np.searchsorted(st,grid)-1,0,None); gv=vol[gi]
for nm,m in [('calm',~gv),('vol',gv)]:
    a=np.abs(pg[m]); print(nm,'mean|pos|',a.mean().round(1),'frac>=90',(a>=90).mean().round(3),'frac<=10',(a<=10).mean().round(3))
me.to_pickle(SP+'/me.pkl')
# minute 60-67 window
w=(t>=3600e6)&(t<4040e6)
print('window 3600-4040', me[w].groupby(['reg','taker'])[['quantity','edge0','mk1','mk30']].sum().round(0))
