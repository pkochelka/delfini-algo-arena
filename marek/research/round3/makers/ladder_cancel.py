"""Ladder cancel law. Exposure on a 5 ms grid; each cancel row is attributed to the state 25 ms earlier (decision time).
Tables: hazard by touch distance (rel) x regime; in the near zone (rel >= -1) by S distance d = (S - price)*side
(>0: S still beyond the order on the safe side) and by mv since insert; lifetimes; fill share."""
import numpy as np, pandas as pd
from s_load import with_s
o = with_s()
q = pd.read_pickle('cache/quotes.pkl'); qt = q.t_us.to_numpy(); qb = q.bid.to_numpy().astype(float); qa = q.ask.to_numpy().astype(float)
St = o.t_us.to_numpy(); Sv = o.S.to_numpy()
def at(T, V, t): return V[np.clip(np.searchsorted(T, t, 'right') - 1, 0, len(T) - 1)]
lad = o[o.lad & o.st.isin(['calm', 'vol'])]
step, LAG = 5000, 25000
out = []
for reg in ('calm', 'vol'):
    s = lad[(lad.st == reg)]
    t0 = s.t_us.to_numpy(); t1 = s.end_t_us.fillna(6e9).to_numpy().astype(np.int64)
    canc = (s.end == 'cancelled').to_numpy(); sg = s.sg.to_numpy(); pr = s.price.to_numpy(); S0 = s.S.to_numpy()
    n = ((t1 - t0)//step).astype(int) + 1
    idx = np.repeat(np.arange(len(s)), n); off = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    tt = t0[idx] + off*step
    rel = (pr[idx] - np.where(sg[idx] > 0, at(qt, qb, tt), at(qt, qa, tt))) * sg[idx]
    d = (at(St, Sv, tt) - pr[idx]) * sg[idx]; mv = -(at(St, Sv, tt) - S0[idx]) * sg[idx]
    tc = np.maximum(t1 - LAG, t0)
    relc = (pr - np.where(sg > 0, at(qt, qb, tc), at(qt, qa, tc))) * sg
    dc = (at(St, Sv, tc) - pr) * sg; mvc = -(at(St, Sv, tc) - S0) * sg
    age = off*step/1e6; agec = (tc - t0)/1e6
    E = pd.DataFrame(dict(rel=np.clip(rel, -9, 0), d=d, mv=mv, old=age > 0.05)); C = pd.DataFrame(dict(rel=np.clip(relc, -9, 0), d=dc, mv=mvc, old=agec > 0.05))[canc]
    h = (C.groupby('rel').size() / (E.groupby('rel').size()*step/1e6)).round(2)
    print(reg, 'hazard by touch distance (decision state):', h.to_dict())
    for var, bins in (('d', [-20, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 20]), ('mv', [-20, -1, 0, 0.5, 1, 2, 3, 4, 6, 20])):
        e = E[(E.rel >= -1) & E.old]; c = C[(C.rel >= -1) & C.old]
        he = (pd.cut(c[var], bins).value_counts(sort=False) / (pd.cut(e[var], bins).value_counts(sort=False)*step/1e6)).round(2)
        print(f'   near zone (rel>=-1) hazard by {var}:', {str(k): v for k, v in he.items()})
    life = (t1 - t0)/1e6
    out.append(dict(regime=reg, n=len(s), filled=(s.end == 'filled').mean(), life_mean=life[canc].mean(),
                    p10=np.percentile(life[canc], 10), p50=np.percentile(life[canc], 50), p90=np.percentile(life[canc], 90), p99=np.percentile(life[canc], 99)))
print(pd.DataFrame(out).round(3).to_string())
