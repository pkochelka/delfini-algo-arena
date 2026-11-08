"""TWAP: signed S move before/during/after, residual (non-impact) part, and trigger."""
import numpy as np, pandas as pd
from common import *
e=pd.read_csv('twap_episodes.csv'); x=L('round4','strue').sp.to_numpy(); r=np.nan_to_num(np.load('cache/round4_sresid05.npy')); cr=np.r_[0,np.cumsum(r)]
v=regime_1s('round4')
def S(t): return np.interp(t,np.arange(len(x))*0.1,x)
def R(t): return np.interp(t,np.arange(len(cr))*0.5,cr)
sg=np.where(e.side=='B',1,-1); t0=e.t0.to_numpy(); t1=e.t1.to_numpy()+0.5
rows=dict(
 pre30=sg*(S(t0)-S(t0-30)), pre5=sg*(S(t0)-S(t0-5)), during=sg*(S(t1)-S(t0)),
 post10=sg*(S(t1+10)-S(t1)), post30=sg*(S(t1+30)-S(t1)), post60=sg*(S(t1+60)-S(t1)), post120=sg*(S(t1+120)-S(t1)),
 res_pre5=sg*(R(t0)-R(t0-5)), res_during=sg*(R(t1)-R(t0)), res_post30=sg*(R(t1+30)-R(t1)))
D=pd.DataFrame(rows); print('signed (TWAP side) S moves, mean ± se over 37 episodes:')
for c in D: print(f'  {c:11s} {D[c].mean():+7.2f} ± {D[c].std()/np.sqrt(len(D)):.2f}')
print('during per child', (D.during/e.n).mean().round(3), ' expected impact per child (calm c=.0166*160^.616*kept)', round(0.0166*160**0.616,3))
print('vol frac at start',v[t0.astype(int)].mean().round(2),' during',np.mean([v[int(a):int(b)].mean() for a,b in zip(t0,t1)]).round(2))
# does TWAP side correlate with the drift sign measured in vol segments etc.? corr of n with |during|
print('corr(n, during)',np.corrcoef(e.n,D.during)[0,1].round(2))
# start-time pattern: start relative to shocks
on=shocks('round4').onset.to_numpy(); dd=[(a-on[on<=a].max()) if (on<=a).any() else np.nan for a in t0]; print('time since last shock at TWAP start',np.round(dd,0).tolist())
