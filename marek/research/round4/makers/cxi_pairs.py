"""Cancel row -> maker insert pairs at exact small dt (1..5 us), classified by the 15-lot agent / ladder / other."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, '/Users/marek/Documents/delfini-algo-arena')
from analysis.cache import load_tape
for rnd in ['round3', 'round4']:
    tape = load_tape(f'/Users/marek/Documents/delfini-algo-arena/data/{rnd}.csv', verbose=False)
    ev = tape.events
    t = ev.t_us.to_numpy(); op = ev.op.astype(str).to_numpy(); lf = ev.lifespan.astype(str).to_numpy()
    isC = op == 'C'; isM = (op == 'I') & (lf == 'G')
    cnt = {}
    nxt_dt = t[1:] - t[:-1]
    for d in [1, 2, 3, 4, 5, 10]:
        # any cancel followed by maker insert at exactly d us (not necessarily adjacent rows)
        tc = t[isC]; tm = t[isM]
        cnt[d] = np.isin(tc + d, tm).sum()
    print(rnd, 'cancel -> maker insert at +d us:', cnt)
    if rnd == 'round4':
        o = pd.read_pickle('cache/orders.pkl'); a = pd.read_pickle('cache/r15_v2.pkl'); ch = pd.read_pickle('cache/chains.pkl')
        tc = set(t[isC].tolist())
        m = o[~o.ioc & o.t_us.sub(1).isin(tc)]
        lab = np.where(m.index.isin(a.index[a.link == 'cancel']), 'agent', np.where(ch.L.reindex(m.index) >= 2, 'ladder', 'other'))
        print(' r4 cancel->insert 1us pairs by class:', pd.Series(lab).value_counts().to_dict())
