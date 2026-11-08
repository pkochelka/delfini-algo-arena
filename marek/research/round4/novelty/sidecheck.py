"""Cross-check flow's side law with causal S: s_filt / s_hat at the PREVIOUS insert row (no own-kick leak). r3 control."""
import numpy as np, pandas as pd
from scipy.optimize import minimize
from common import *
for n in ['round3','round4']:
    o=L(n,'orders'); s=pd.read_parquet(f'/Users/marek/Documents/delfini-algo-arena/marek/research/{n}/s_dynamics/s_path.parquet',columns=['s_hat','s_filt'])
    sh=s.s_hat.to_numpy(); sf=s.s_filt.to_numpy()
    idx=np.where(o.ioc.to_numpy())[0]; k=o.iloc[idx]
    if n=='round4': keep=~L('round4','takers_tw').tw.to_numpy()
    else: keep=np.ones(len(k),bool)
    bad=np.zeros(len(k),bool)
    for a in shocks(n).onset: bad|=((k.t>a-1)&(k.t<a+8)).to_numpy()
    m=keep&~bad&(k.t>25).to_numpy()
    mid=((k.pbid+k.pask)/2).to_numpy(); y=(k.side==0).to_numpy().astype(float)
    v=regime_1s(n)[k.t.astype(int)]
    for nm,S in [('s_hat own row',sh[idx]),('s_hat prev row',sh[idx-1]),('s_filt prev row',sf[idx-1])]:
        x=S-mid
        for R in [False,True]:
            mm=m&(v==R)&np.isfinite(x)
            A=np.c_[np.ones(mm.sum()),x[mm]]; yy=y[mm]
            f=lambda b: np.sum(np.logaddexp(0,A@b)-yy*(A@b))
            r=minimize(f,np.zeros(2),method='BFGS'); p0=yy.mean(); l0=-mm.sum()*(-(p0*np.log(p0)+(1-p0)*np.log(1-p0)))
            print(f'{n} {nm:16s} {"vol " if R else "calm"} n={mm.sum()} slope={r.x[1]:+.3f} dLL={-r.fun-l0:7.1f} sd(S-mid)={x[mm].std():.2f}')
