"""Ladder hazard vs drift of S (and of own touch) since the last same-side ladder start; and vs the number of live
ladder orders (from makers' cancel times not available -> use orders end_t)."""
import numpy as np, pandas as pd
from pathlib import Path
D = Path(__file__).parent; C = D / 'cache'
r = pd.read_pickle(C / 'rows.pkl'); ins = r[r.op == 0].reset_index(drop=True)
S = pd.read_parquet(D.parent / 's_dynamics/s_path.parquet'); L = pd.read_pickle(C / 'ladders.pkl')
o = pd.read_pickle(C / 'orders.pkl'); mem = np.load(C / 'ladder_member.npy')
ti = ins.t.to_numpy(); s = S.s_hat.to_numpy(); vol = S.vol.to_numpy(); shock = S.shock.to_numpy()
W = 0.01; g = np.arange(0, 6000, W) + W; k = np.clip(np.searchsorted(ti, g) - 1, 0, None)
Sg = s[k]; Vg = vol[k]; Hg = shock[k].copy()
for oo in pd.read_csv(D / 'shock_events.csv').onset: Hg[int((oo - 1) / W):int((oo + 8) / W)] = True
kr = np.clip(np.searchsorted(r.t.to_numpy(), g) - 1, 0, None)
# live ladder orders count per side at grid times
end_t = o.end_t_us.to_numpy() / 1e6
is_l = mem[:, 0] >= 0
res = []
for side, col in [(0, 'bid_after'), (1, 'ask_after')]:
    tp = r[col].to_numpy()[kr]
    li = np.flatnonzero(is_l & (ins.side.to_numpy() == side))
    st = np.sort(ti[li]); en = np.sort(end_t[li])
    live = np.searchsorted(st, g) - np.searchsorted(en, g)
    lt = np.sort(L.t0[L.side == side].to_numpy()); j = np.searchsorted(lt, g) - 1
    has = j >= 0; t_last = np.where(has, lt[np.maximum(j, 0)], 0)
    kl = np.clip(np.searchsorted(ti, t_last) - 1, 0, None)
    driftS = np.abs(Sg - s[kl]); tpl = r[col].to_numpy()[np.clip(np.searchsorted(r.t.to_numpy(), t_last) - 1, 0, None)]
    driftT = np.abs(tp - tpl)
    y = np.r_[np.bincount((lt / W).astype(int), minlength=len(g))[1:len(g)], 0]
    res.append(pd.DataFrame({'y': y, 'vol': Vg, 'H': Hg, 'dS': driftS, 'dT': driftT, 'live': live, 'age': g - t_last}))
X = pd.concat(res); X = X[~X.H & (X.age < 1e3)]
pd.set_option('display.width', 220)
for v in [False, True]:
    Y = X[X.vol == v]; print(f'==== vol={v}')
    print(' |S drift since last same-side ladder|:'); print(Y.groupby(pd.cut(Y.dS, [-.01, .1, .3, .6, 1, 1.5, 2, 3, 5, 100])).y.agg(['mean', 'size']).assign(rate=lambda d: d['mean'] / W).round(3).T.to_string())
    print(' |touch drift since last same-side ladder|:'); print(Y.groupby(pd.cut(Y.dT, [-.5, .5, 1.5, 2.5, 3.5, 5.5, 100])).y.agg(['mean', 'size']).assign(rate=lambda d: d['mean'] / W).round(3).T.to_string())
    print(' live ladder orders same side:'); print(Y.groupby(pd.cut(Y.live, [-1, 0, 1, 2, 3, 5, 8, 12, 20, 1000])).y.agg(['mean', 'size']).assign(rate=lambda d: d['mean'] / W).round(3).T.to_string())
