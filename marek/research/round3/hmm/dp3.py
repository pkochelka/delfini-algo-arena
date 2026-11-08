"""Same DP but valued at the edge at arrival (no S-path hindsight), volatile IOC events; greedy vs optimal; then
threshold policies (take a lot only if its edge >= thr) to see if room-aware selectivity helps without hindsight."""
import numpy as np, pandas as pd
C='marek/research/round3/ideas/ceiling/_cache/'
st=pd.read_parquet(C+'ceil_stale.parquet'); ev=pd.read_parquet(C+'ceil_events.parquet')
vs=pd.read_csv('marek/research/round3/s_dynamics/vol_segments.csv'); sh=pd.read_csv('marek/research/round3/flow/shock_events.csv')
def isvol(us):
    x=us/1e6; v=np.zeros(len(x),bool)
    for a,b in zip(vs.start,vs.end): v|=(x>=a)&(x<b)
    for o in sh.onset: v&=~((x>=o-1)&(x<o+20))
    return v
ev=ev[isvol(ev.t_us.values)]; g=st.groupby(['t_us','side']); E=[]
for tu,sd in zip(ev.t_us,ev.side):
    d=g.get_group((tu,sd)); dr=1 if sd=='A' else -1
    E.append((tu,dr,np.sort(np.repeat(d.edge_arr.values,d.rem_arr.values.astype(int)))[::-1][:200]))
cap=100; N=2*cap+1; V=np.zeros(N)
for tu,dr,vals in reversed(E):
    cum=np.concatenate([[0],np.cumsum(vals)]); new=V.copy()
    for q in range(1,len(cum)):
        c=np.full(N,-np.inf)
        if dr>0: c[:N-q]=cum[q]+V[q:]
        else: c[q:]=cum[q]+V[:N-q]
        new=np.maximum(new,c)
    V=new
print('DP edge-at-arrival', V[cap]/1e3)
for thr in [0,0.5,0.75,1.0,1.25,1.5,2.0,2.5]:
  for thr_add in [None]:
    pos=0; gv=0; gl=0
    for tu,dr,vals in E:
        room=cap-dr*pos; v=vals[vals>=thr]; q=min(room,len(v)); gv+=v[:q].sum(); gl+=q; pos+=dr*q
    print(f'greedy thr {thr}: {gv/1e3:.1f}k lots {gl}')
# split threshold: reducing lots (pos opposite) at thr_r, adding lots at thr_a
for tr_ in [0.5,1.0]:
  for ta in [1.0,1.5,2.0,3.0]:
    pos=0; gv=0
    for tu,dr,vals in E:
        red=max(0,-dr*pos)  # lots that reduce |pos|
        v=vals[vals>=tr_]; q1=min(red,len(v)); gv+=v[:q1].sum(); pos+=dr*q1
        rest=vals[q1:]; v2=rest[rest>=ta]; room=cap-dr*pos; q2=min(room,len(v2)); gv+=v2[:q2].sum(); pos+=dr*q2
    print(f'split red>={tr_} add>={ta}: {gv/1e3:.1f}k')
