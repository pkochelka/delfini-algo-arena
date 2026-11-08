"""Room-constrained hindsight DP in volatile: stale IOC events vs residual (limit-quote) events, both valued at the
1 s markout (S(t+1) - price), +-100. Answers: does adding the residual events raise the ceiling much?"""
import numpy as np, pandas as pd
C='marek/research/round3/ideas/ceiling/_cache/'
st=pd.read_parquet(C+'ceil_stale.parquet'); ev=pd.read_parquet(C+'ceil_events.parquet')
t=pd.read_parquet(C+'ceil_takers.parquet'); s=pd.read_parquet('marek/research/round3/s_dynamics/s_path.parquet')
vs=pd.read_csv('marek/research/round3/s_dynamics/vol_segments.csv'); sh=pd.read_csv('marek/research/round3/flow/shock_events.csv')
stt=s.t_us.values; sv=s.s_hat.values
S_at=lambda us: sv[np.clip(np.searchsorted(stt,us,side='right')-1,0,len(sv)-1)]
def isvol(us):
    x=us/1e6; v=np.zeros(len(x),bool)
    for a,b in zip(vs.start,vs.end): v|=(x>=a)&(x<b)
    for o in sh.onset: v&=~((x>=o-1)&(x<o+20))
    return v
E=[]  # (t_us, dir +1 buy/-1 sell, values sorted desc)
ev=ev[isvol(ev.t_us.values)]
g=st.groupby(['t_us','side'])
for tu,sd in zip(ev.t_us,ev.side):
    d=g.get_group((tu,sd)); S1=S_at(np.array([tu+5000+1_000_000]))[0]
    dr=1 if sd=='A' else -1
    v=dr*(S1-d.price.values)
    E.append((tu,dr,np.sort(np.repeat(v,d.rem_arr.values.astype(int)))[::-1][:200],'ioc'))
t=t[isvol(t.t_us.values)&((t.q-t.filled)>0)]
for tu,sd,lim,r in zip(t.t_us,t.side,t.lim,t.q-t.filled):
    S1=S_at(np.array([tu+1_000_000]))[0]; dr=-1 if sd=='B' else 1  # buy taker -> we sell
    E.append((tu,dr,np.full(min(int(r),200),dr*(S1-lim)),'res'))
E.sort(key=lambda x:x[0])
def dp(events,cap=100):
    N=2*cap+1; V=np.zeros(N)
    for tu,dr,vals,_ in reversed(events):
        cum=np.concatenate([[0],np.cumsum(vals)]); new=V.copy()
        for q in range(1,len(cum)):
            c=np.full(N,-np.inf)
            if dr>0: c[:N-q]=cum[q]+V[q:]
            else: c[q:]=cum[q]+V[:N-q]
            new=np.maximum(new,c)
        V=new
    return V[cap]
print('ioc only', dp([e for e in E if e[3]=='ioc'])/1e3)
print('res only', dp([e for e in E if e[3]=='res'])/1e3)
print('both    ', dp(E)/1e3)

# ---- forward pass of the optimal policy + greedy comparison (IOC only)
ioc=[e for e in E if e[3]=='ioc']
cap=100; N=2*cap+1
Vs=[None]*(len(ioc)+1); Vs[-1]=np.zeros(N)
for i in range(len(ioc)-1,-1,-1):
    tu,dr,vals,_=ioc[i]; V=Vs[i+1]; cum=np.concatenate([[0],np.cumsum(vals)]); new=V.copy()
    for q in range(1,len(cum)):
        c=np.full(N,-np.inf)
        if dr>0: c[:N-q]=cum[q]+V[q:]
        else: c[q:]=cum[q]+V[:N-q]
        new=np.maximum(new,c)
    Vs[i]=new
pos=0; rows=[]
for i,(tu,dr,vals,_) in enumerate(ioc):
    cum=np.concatenate([[0],np.cumsum(vals)]); best=(Vs[i+1][pos+cap],0)
    room=cap-dr*pos
    for q in range(1,min(len(cum)-1,room)+1):
        v=cum[q]+Vs[i+1][pos+dr*q+cap]
        if v>best[0]+1e-9: best=(v,q)
    q=best[1]
    rows.append((tu,dr,pos,room,min(room,len(vals)),q,cum[q], vals[0]))
    pos+=dr*q
R=pd.DataFrame(rows,columns=['t','dr','pos','room','avail','q','val','v1'])
print('opt path value',R.val.sum()/1e3,'lots',R.q.sum(), 'share of avail taken',R.q.sum()/R.avail.sum())
# greedy: take everything available up to room
pos=0; gv=0; gl=0
for tu,dr,vals,_ in ioc:
    room=cap-dr*pos; q=min(room,len(vals)); gv+=vals[:q].sum(); gl+=q; pos+=dr*q
print('greedy value',gv/1e3,'lots',gl)
R['part']=np.where(R.q==0,'skip',np.where(R.q<R.avail,'partial','full'))
print(R.groupby('part').agg(n=('q','size'),room=('room','mean'),avail=('avail','mean'),q=('q','mean'),v1=('v1','mean')))
R.to_parquet('marek/research/round3/hmm/dp_opt_path.parquet')
