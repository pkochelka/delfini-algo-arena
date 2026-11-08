import pandas as pd, numpy as np
t=pd.read_parquet('marek/research/round3/ideas/ceiling/_cache/ceil_takers.parquet')
s=pd.read_parquet('marek/research/round3/s_dynamics/s_path.parquet')
vs=pd.read_csv('marek/research/round3/s_dynamics/vol_segments.csv')
sh=pd.read_csv('marek/research/round3/flow/shock_events.csv')
ts=t.t_us.values/1e6
vol=np.zeros(len(t),bool)
for a,b in zip(vs.start,vs.end): vol|=(ts>=a)&(ts<b)
shk=np.zeros(len(t),bool)
for o in sh.onset: shk|=(ts>=o-1)&(ts<o+20)
t['reg']=np.where(shk,'shock',np.where(vol,'vol','calm'))
t['r']=t.q-t.filled
sg=np.where(t.side=='B',1,-1)
st=s.t_us.values; sh_=s.s_hat.values
def S_at(us): 
    i=np.searchsorted(st,us,side='right')-1; return sh_[np.clip(i,0,len(sh_)-1)]
t['off']=sg*(t.lim-t.S_before)
for h in [0,0.2,1,3,10,30]:
    t[f'm{h}']=sg*(t.lim-S_at(t.t_us.values+int(h*1e6)+1)) if h>0 else sg*(t.lim-t.S_after)
R=t[t.r>0]
for g,d in R.groupby('reg'):
    w=np.minimum(d.r,100)
    print(g,len(d),'lots(cap100)',w.sum(),' '.join(f'{c}:{(w*d[c]).sum()/w.sum():.2f}' for c in ['m0','m0.2','m1','m3','m10','m30']))
    print('  offset (lim - S_before) quantiles', d.off.quantile([.05,.25,.5,.75,.95]).round(2).tolist(), 'residual median',d.r.median())
# how often is lim == round(S_before +- 2.55) in vol, round(S+-1) calm
for g,d in t.groupby('reg'):
    sgn=np.where(d.side=='B',1,-1)
    for k in [1,2.55]:
        print(g,k,'share lim==round(S+-k)',np.mean(d.lim==np.round(d.S_before+sgn*k)).round(3))
print('--- conditional on book-lag g = sg*(S_before - mid): >0 means S beyond mid in the taker direction')
mid=(t.bb+t.ba)/2
t['g']=sg*(t.S_before-mid)
R=t[(t.r>0)&(t.reg=='vol')]
w=np.minimum(R.r,100)
for lo,hi in [(-9,-1),(-1,-0.3),(-0.3,0.3),(0.3,1),(1,2),(2,9)]:
    m=(R.g>=lo)&(R.g<hi)
    ww=w[m]
    print(f'g in [{lo},{hi}) n={m.sum()} lots={ww.sum()} '+' '.join(f'{c}:{(ww*R[c][m]).sum()/max(ww.sum(),1):.2f}' for c in ['m0','m1','m3','m10']))
print('NaN mid share', mid.isna().mean())
