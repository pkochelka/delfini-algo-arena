import numpy as np, pandas as pd
from common import *
def segs(v):
    d=np.diff(np.r_[0,v[:6000].astype(int),0]); return list(zip(np.where(d==1)[0],np.where(d==-1)[0]))
for n in ['round2','round3']:
    o=L(n,'orders'); v=regime_1s(n); x=L(n,'strue').sp.to_numpy(); sh=shocks(n).onset.to_numpy()
    def at(t): return x[np.clip((np.asarray(t)*10).astype(int),0,len(x)-1)]
    k=o[o.ioc&(o.filled>0)]; imp=np.where(k.side==0,1,-1)*k.filled**0.62; kt=k.t.to_numpy()
    rows=[]
    for a,b in segs(v):
        if b-a<10: continue
        if np.any((sh>a-5)&(sh<b+1)): continue
        m=(kt>=a)&(kt<b); I=imp[m].sum(); nimb=np.sign(imp[m]).sum()
        pre=at(a)-at(a-20); 
        rows.append((b-a,at(b)-at(a),I,nimb,pre, at(a+5)-at(a)))
    R=pd.DataFrame(rows,columns=['dur','dS','I','nimb','pre20','first5'])
    b=np.sum(R.dS*R.I)/np.sum(R.I**2); res=R.dS-b*R.I
    print(n,'vol segments',len(R),f'dS~I slope={b:.4f} resid sd/sqrt(dur)={np.sqrt(np.mean(res**2/R.dur)):.2f} mean dS={R.dS.mean():+.2f} corr(pre20,dS)={np.corrcoef(R.pre20,R.dS)[0,1]:+.2f} corr(nimb,dS)={np.corrcoef(R.nimb,R.dS)[0,1]:+.2f} mean|first5|={R.first5.abs().mean():.2f}')
