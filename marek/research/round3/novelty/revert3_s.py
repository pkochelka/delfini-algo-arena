import numpy as np, pandas as pd
from common import *
exec(open('revert2.py').read().split('# round1 S control')[0])
for n,src in [('round2','strue'),('round3','strue')]:
    g=L(n,src); x=g.sp.to_numpy(); t=g.t.to_numpy()
    bad=np.zeros(len(t),bool)
    for o in shocks(n).onset: bad|=(t>o-1)&(t<o+30)
    run(n,x,None,bad,f'{n} {src} no regime filter')
