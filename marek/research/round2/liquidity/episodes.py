import sys, importlib.util, numpy as np, pandas as pd
from pathlib import Path
from simulator.simulator import ExchangeSimulator, SimulationConfig
spec=importlib.util.spec_from_file_location('st', sys.argv[1]); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
Cls=[v for v in vars(m).values() if isinstance(v,type) and issubclass(v,m.BaseStrategy) and v is not m.BaseStrategy][0]
S=pd.read_parquet('marek/research/round2/s_dynamics/s_path.parquet'); st=S.t_us.values; sh=S.s_hat.values; sv=S.vol.values
THR=float(sys.argv[3])
class Rec(Cls):
    def __init__(s):
        super().__init__(); s.ep=[]; s.act={1:None,-1:None}
    def on_order_book_change(s, snap, trades):
        super().on_order_book_change(snap, trades)
        t=snap.timestamp_us
        i=np.searchsorted(st,t,side='right')-1
        if i<0: return
        S0=sh[i]
        # market-only stale qty and edge (exclude our own orders: owner unknown in view? use all)
        aq=sum(o.quantity*(S0-o.price) for o in snap.asks if o.price<S0-THR)
        bq=sum(o.quantity*(o.price-S0) for o in snap.bids if o.price>S0+THR)
        aqq=sum(o.quantity for o in snap.asks if o.price<S0-THR)
        bqq=sum(o.quantity for o in snap.bids if o.price>S0+THR)
        for sd,q,e in ((1,aqq,aq),(-1,bqq,bq)):   # sd=+1: buy opportunity
            if q>0 and s.act[sd] is None:
                s.act[sd]=[t,sd,q,e,s.pos,bool(sv[i]),q,e,(s.S if s.S is not None else np.nan)-S0,bool(s.vol),bool(getattr(s,"sg_active",False))]
            elif q>0:
                a=s.act[sd]; a[6]=max(a[6],q); a[7]=max(a[7],e)
            elif q==0 and s.act[sd] is not None:
                a=s.act[sd]; s.ep.append(a+[t,s.pos]); s.act[sd]=None
st_=Rec()
sim=ExchangeSimulator(strategy=st_,config=SimulationConfig(public_depth=20))
r=sim.run(Path('data/round2.csv'))
df=pd.DataFrame(st_.ep,columns=['t0','sd','q0','e0','pos0','vol','qmax','emax','serr','dvol','sg','t1','pos1'])
df.to_pickle(sys.argv[2]); print(r.pnl, len(df))
