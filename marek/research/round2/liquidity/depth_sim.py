import numpy as np, sys
from pathlib import Path
from simulator.simulator import ExchangeSimulator, SimulationConfig
from simulator.strategy import BaseStrategy
class Rec(BaseStrategy):
    def __init__(s):
        super().__init__(); s.rows=[]; s.nxt=0; s.t0=None; s.tv=0; s.tn=0
    def on_order_book_change(s, snap, trades):
        t=snap.timestamp_us
        if s.t0 is None: s.t0=t
        for tr in trades: s.tv+=tr.quantity; s.tn+=1
        if t-s.t0>=s.nxt:
            b=snap.bids; a=snap.asks
            def agg(side):
                d={}
                for o in side: d[o.price]=d.get(o.price,0)+o.quantity
                return d
            bd=agg(b); ad=agg(a)
            bb=max(bd) if bd else np.nan; ba=min(ad) if ad else np.nan
            row=[(t-s.t0)/1e6,bb,ba,bd.get(bb,0),ad.get(ba,0),
                 sum(v for p,v in bd.items() if p>=bb-2),sum(v for p,v in ad.items() if p<=ba+2),
                 sum(bd.values()),sum(ad.values()),len(b)+len(a),s.tv,s.tn]
            s.rows.append(row); s.tv=0; s.tn=0
            s.nxt+=100_000
st=Rec()
sim=ExchangeSimulator(strategy=st,config=SimulationConfig(public_depth=20))
sim.run(Path(sys.argv[1]))
np.save(sys.argv[2],np.array(st.rows,dtype=float))
print(len(st.rows))
