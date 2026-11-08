import os, numpy as np, pandas as pd
R='/Users/marek/Documents/delfini-algo-arena/marek/research/'
HERE='/Users/marek/Documents/delfini-algo-arena/marek/research/round4/novelty/'
def regime_1s(n, force_flow=False):
    v=np.zeros(6001,bool)
    if n=='round2':
        s=pd.read_csv(R+'round2/s_dynamics/vol_segments.csv')
        for a,b in zip(s.start,s.end): v[int(np.floor(a)):int(np.ceil(b))]=True
        return v
    f=R+n+'/flow/regimes_1s.csv'
    if os.path.exists(f) and (n=='round3' or force_flow or True):
        s=pd.read_csv(f)
        if 'regime' in s:
            for a,b,r in zip(s.start,s.end,s.regime):
                if r=='volatile': v[int(a):int(b)]=True
            return v
    p=HERE+f'cache/{n}_volproxy.npy'
    if os.path.exists(p): v[:6000]=np.load(p); v[6000]=v[5999]; return v
    raise FileNotFoundError('no regime for '+n)
LEAD_ON=[167.5,462,661.5,811.5,1007,1175,1391,1541,1737.5,1916,2168,2355.5,2538,2686.5,2940,3143,3384.5,3565.5,3833,4076,4259.5,4467.5,4711.5,5008.5,5243,5422,5630,5799]
def shocks(n):
    f=R+n+'/flow/shock_events.csv'
    if os.path.exists(f): return pd.read_csv(f)
    if n=='round4': return pd.DataFrame(dict(onset=LEAD_ON))
    raise FileNotFoundError(f)
def L(n,what): return pd.read_parquet(HERE+f'cache/{n}_{what}.parquet')
