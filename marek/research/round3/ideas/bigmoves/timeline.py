"""Per-shock event timeline around the jump: cancels (side, stale?), takers (levels swept), maker inserts at new level."""
import sys, numpy as np, pandas as pd
from common import *
name = sys.argv[1]; which = [int(x) for x in sys.argv[2].split(',')] if len(sys.argv) > 2 else None
tp, o, tr = load(name); ev = tp.events
S = s_path(name); sh = shocks(name, S)
oid2 = o.set_index('order_id')
evt = ev.t_us.values * 1e-6
for k, s in enumerate(sh):
    if which and k not in which: continue
    d = 1 if s['J'] > 0 else -1
    Sn = s.get('S_new')
    i0 = np.searchsorted(evt, s['tj'] - 0.12); i1 = np.searchsorted(evt, s['tj'] + 0.08)
    print(f"=== shock {k} onset {s['onset']:.2f} tj {s['tj']:.4f} J {s['J']:+.1f} S_new {Sn}")
    for i in range(i0, i1):
        r = ev.iloc[i]; t = evt[i] - s['tj']
        if r.op == 'C':
            if r.order_id in oid2.index:
                q = oid2.loc[r.order_id]; sd = q.side; px = q.price
                stale = (sd == 'A' and d > 0 and px < Sn) or (sd == 'B' and d < 0 and px > Sn) if Sn else None
                print(f"  {t*1e3:+8.2f}ms C {sd} {px} stale={stale}   bb {r.bid} ba {r.ask}")
        else:
            kind = 'IOC' if r.lifespan == 'F' else 'G'
            print(f"  {t*1e3:+8.2f}ms {kind} {r.side} {int(r.volume)}@{int(r.price)}   bb {r.bid} ba {r.ask}")
