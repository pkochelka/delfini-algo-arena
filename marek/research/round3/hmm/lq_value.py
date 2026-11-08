"""Value of a resting quote (behind the market queue) at P = L + k, L = the taker's own limit, per regime.
Fill of our Q-lot quote = min(Q, original fill volume beyond P + residual). Edge vs S after the kick and at +1 s."""
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
st=s.t_us.values; sh_=s.s_hat.values
def S_at(us):
    i=np.searchsorted(st,us,side='right')-1; return sh_[np.clip(i,0,len(sh_)-1)]
S1=S_at(t.t_us.values+1_000_000); S3=S_at(t.t_us.values+3_000_000)
sg=np.where(t.side=='B',1,-1)
fills=[[(int(a),int(b)) for a,b in (x.split(':') for x in f.split(';'))] if f else [] for f in t.fills]
r=(t.q-t.filled).values
Q=100
for reg in ['vol','calm']:
    m=(t.reg==reg).values
    base_k = 2.55 if reg=='vol' else 1.0
    for mode in ['L','S']:
        for k in ([-2,-1,0] if mode=='L' else [x for x in (np.arange(1.0,3.6,0.25) if reg=='vol' else np.arange(0.25,1.6,0.25))]):
            lots=e0=e1=e3=0.0; n=0
            for i in np.nonzero(m)[0]:
                if mode=='L': P=t.lim.values[i]+sg[i]*k
                else: P=np.ceil(t.S_before.values[i]+k) if sg[i]>0 else np.floor(t.S_before.values[i]-k)
                if sg[i]*(t.lim.values[i]-P)<0: continue  # beyond the limit
                beyond=sum(q for px,q in fills[i] if sg[i]*(px-P)>0)
                f=min(Q,beyond+r[i])
                if f<=0: continue
                n+=1; lots+=f
                e0+=f*sg[i]*(P-t.S_after.values[i]); e1+=f*sg[i]*(P-S1[i]); e3+=f*sg[i]*(P-S3[i])
            print(f'{reg} {mode} k={k:5.2f} fills={n:5d} lots={lots:8.0f} e0={e0/max(lots,1):5.2f} e1={e1/max(lots,1):5.2f} e3={e3/max(lots,1):5.2f}  val1={e1/1e3:7.1f}k')
