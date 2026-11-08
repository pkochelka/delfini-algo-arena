import numpy as np, pandas as pd
from common import *
s=pd.read_csv('../s_dynamics/vol_segments.csv'); x=L('round4','strue').sp.to_numpy(); S=lambda t: np.interp(t,np.arange(len(x))*0.1,x)
r=np.nan_to_num(np.load('cache/round4_sresid05.npy')); cr=np.r_[0,np.cumsum(r)]; R=lambda t: np.interp(t,np.arange(len(cr))*0.5,cr)
o=L('round4','orders'); k=o[o.ioc]; m=o[~o.ioc]
s['f']=np.abs(s.start-np.round(s.start)); s['int']=s.f<0.05
rows=[]
for _,e in s.iterrows():
    a,b=e.start,e.end
    tr=np.abs(R(b)-R(a)); path=np.sum(np.abs(np.diff(R(np.arange(a,b,1.0)))))
    kr=((k.t>=a)&(k.t<b)).sum()/(b-a)
    rows.append((e.start,e.dur,e.int,tr,path/(b-a),kr, S(a+5)-S(a)))
D=pd.DataFrame(rows,columns=['start','dur','int','net_resid','resid_speed','taker_rate','dS_first5'])
print(D.groupby('int').agg(['mean','count']).round(2).T.to_string())
print(D.sort_values('start').round(2).to_string())
