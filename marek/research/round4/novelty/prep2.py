import sys, numpy as np
sys.path.insert(0,'/Users/marek/Documents/delfini-algo-arena')
from analysis.cache import load_tape
for n in sys.argv[1:]:
    t=load_tape(f'/Users/marek/Documents/delfini-algo-arena/data/{n}.csv'); ev=t.events
    np.savez(f'cache/{n}_top.npz', t=(ev.t_us.to_numpy()-ev.t_us.iloc[0])/1e6, bid=ev.bid.to_numpy(), ask=ev.ask.to_numpy(), bq=ev.bid_qty.to_numpy(), aq=ev.ask_qty.to_numpy())
