import numpy as np, pandas as pd
from common import *
for n in ['round2','round3']:
    o=L(n,'orders'); c=L(n,'cancels')
    for nm,tt in [('taker',o.t[o.ioc]),('maker',o.t[~o.ioc]),('cancel',c.t)]:
        x=np.bincount((tt.to_numpy()*10).astype(int),minlength=60000)[:60000].astype(float); x-=x.mean()
        P=np.abs(np.fft.rfft(x))**2; f=np.fft.rfftfreq(60000,0.1)
        m=(f>1/300)&(f<4.9)
        Pm=P[m]; fm=f[m]; z=Pm/np.median(Pm)
        top=np.argsort(z)[-4:][::-1]
        print(n,nm,'top spectral peaks (period s, power/median):',[(round(1/fm[i],3),round(z[i],1)) for i in top])
    # S mod 100 / 10 structure
    s=L(n,'strue').sp.to_numpy()
    for M in [10,50,100]:
        h=np.histogram(np.mod(s,M),bins=10,range=(0,M))[0]; print(f'   S mod {M} occupancy (deciles):',(h/h.sum()).round(3).tolist())
