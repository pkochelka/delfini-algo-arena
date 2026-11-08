"""4-agent test: at each burst start, live chain-ladder count c at (a) chosen levels, (b) levels skipped inside the
burst span, (c) band levels [-7,-2] outside the span."""
import numpy as np, pandas as pd
from collections import Counter
o = pd.read_pickle('cache/chains.pkl')
lad = o[(o.L >= 2) & (o.L <= 6) & ~o.ioc.astype(bool)].copy()
lad['end_t'] = lad.end_t_us.fillna(6e9)
t = lad.t_us.to_numpy(); e = lad.end_t.to_numpy(); s = lad.sg.to_numpy(); pr = lad.price.to_numpy()
grp = lad.groupby('chain')
A, B, C = Counter(), Counter(), Counter()
for i, (cid, ch) in enumerate(grp):
    if i % 2: continue
    r = ch.iloc[0]; t0 = r.t_us
    tb = r.bid if r.sg > 0 else r.ask
    if np.isnan(tb): continue
    j0 = np.searchsorted(t, t0 - 120e6); j1 = np.searchsorted(t, t0)
    live = (e[j0:j1] > t0) & (s[j0:j1] == r.sg)
    cnt = Counter(((pr[j0:j1][live] - tb) * r.sg).astype(int))
    rels = ((ch.price.to_numpy() - tb) * r.sg).astype(int)
    for x in rels: A[cnt.get(x, 0)] += 1
    lo, hi = rels.min(), rels.max()
    for x in range(lo, hi + 1):
        if x not in rels: B[cnt.get(x, 0)] += 1
    for x in range(-7, -1):
        if x < lo or x > hi: C[cnt.get(x, 0)] += 1
print('chosen levels: count of live ladder orders already there', sorted(A.items()))
print('skipped levels inside span', sorted(B.items()))
print('band levels outside span', sorted(C.items()))
