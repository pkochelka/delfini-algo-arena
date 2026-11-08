import numpy as np, pandas as pd
R='/Users/marek/Documents/delfini-algo-arena/marek/research/'
def regime_1s(n):
    v=np.zeros(6001,bool)
    if n=='round2':
        s=pd.read_csv(R+'round2/s_dynamics/vol_segments.csv')
        for a,b in zip(s.start,s.end): v[int(np.floor(a)):int(np.ceil(b))]=True
    else:
        s=pd.read_csv(R+'round3/flow/regimes_1s.csv')
        for a,b,r in zip(s.start,s.end,s.regime):
            if r=='volatile': v[a:b]=True
    return v
def shocks(n):
    f=R+('round2' if n=='round2' else 'round3')+'/flow/shock_events.csv'
    return pd.read_csv(f)
def L(n,what): return pd.read_parquet(f'/Users/marek/Documents/delfini-algo-arena/marek/research/round3/novelty/cache/{n}_{what}.parquet')
