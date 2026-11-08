"""What triggers a ladder? Time since last event of various kinds (taker, trade, cancel, any row) at ladder start."""
import numpy as np, pandas as pd
from pathlib import Path
D = Path(__file__).parent; C = D / 'cache'
r = pd.read_pickle(C / 'rows.pkl'); L = pd.read_pickle(C / 'ladders.pkl')
L = L[L.steps.apply(lambda x: all(s == 1 for s in x))]
t0 = L.t0.to_numpy(); us0 = np.rint(t0 * 1e6).astype(np.int64)
tt = np.rint(r.t.to_numpy() * 1e6).astype(np.int64)
for nm, sel in [('taker', (r.op == 0) & r.ioc), ('cancel', r.op == 1), ('maker', (r.op == 0) & ~r.ioc), ('any', r.op >= 0)]:
    x = tt[sel.to_numpy()]
    k = np.searchsorted(x, us0) - 1; d = us0 - x[np.maximum(k, 0)]
    h = np.bincount(np.minimum(d, 5000) // 100)  # 0.1 ms bins up to 5 ms
    print(nm, 'dt since last (0.1ms bins, first 20):', h[:20].tolist())
    print('   exact us values top:', pd.Series(d[d < 20000]).value_counts().head(8).to_dict())
# same-side vs opposite of last taker
k = np.searchsorted(tt[((r.op == 0) & r.ioc).to_numpy()], us0) - 1
tk = r[(r.op == 0) & r.ioc]
