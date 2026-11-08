"""Ceiling angle, step 6: how much of the reachable stale-quote value r3_best had room for, given its own position path."""
import sys, pathlib
import numpy as np, pandas as pd
H = pathlib.Path(__file__).parent
ev = pd.read_parquet(H / '_cache/ceil_events.parquet')
st = pd.read_parquet(H / '_cache/ceil_stale.parquet')
st = st[(st.rem_arr > 0) & (st.edge_arr > 0.5)]
E = np.load(sys.argv[1]); et, ep = E['t_us'], E['pos']
j = np.maximum(np.searchsorted(et, ev.t_us.values, side='right') - 1, 0)
ev['pos'] = ep[j]
ev['room'] = np.where(ev.side == 'A', 100 - ev.pos, 100 + ev.pos).clip(0, 200)
grp = {k: g for k, g in st.groupby(['t_us', 'side'])}
vr = []
for t, s, room in zip(ev.t_us, ev.side, ev.room):
    g = grp[(t, s)].sort_values('edge_arr', ascending=False)
    q = g.rem_arr.values; e = g.edge_arr.values; cq = np.cumsum(q)
    take = np.clip(room - (cq - q), 0, q)
    vr.append(float((take * e).sum()))
ev['Vroom'] = vr
pd.set_option('display.width', 200)
g = ev.groupby('reg')[['V100', 'V200', 'Vroom', 'V1000']].sum().round()
g['room0_share'] = ev.groupby('reg').apply(lambda x: (x.room <= 5).mean()).round(3)
g['V100_room0'] = ev[ev.room <= 5].groupby('reg').V100.sum().round()
print(g)
