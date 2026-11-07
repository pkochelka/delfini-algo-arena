"""Maker arrivals: size tail vs rounded lognormal, size/side by regime & around shocks, side choice vs imbalance/trend, placement by regime."""
import numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import norm, kstest
D = Path(__file__).parent; C = D / 'cache'
r = pd.read_pickle(C / 'rows.pkl'); ev = pd.read_csv(D / 'shock_events.csv')
hi = np.load(C / 'hmm_taker_gamma2.npy')[:, 1] > 0.5
tt = r.t.to_numpy(); mid = ((r.bid_after + r.ask_after) / 2).ffill().to_numpy()
idx = np.where((r.op == 0) & ~r.ioc)[0]; m = r.iloc[idx].copy()
m['hi'] = hi[m.t.astype(int).clip(0, 5999)]
rel = np.full(len(m), np.inf)
for o in ev.onset:
    d = m.t.to_numpy() - o; sel = (d > -5) & (d < 20); rel[sel] = d[sel]
m['rel'] = rel
mu, sd = 2.6569, 1.8028
v = m.vol.to_numpy()
print('maker size tail: observed vs rounded-lognormal(2.657,1.803) expected')
for thr in [100, 1000, 5000, 10000, 20000, 40000, 60000]:
    p = 1 - norm.cdf((np.log(thr + 0.5) - mu) / sd)
    print(f'  >= {thr:6d}+: obs {np.sum(v > thr):7d}  exp {len(v) * p:9.1f}')
print('top 10 maker sizes', np.sort(v)[-10:])
lv = np.log(v[v >= 3])  # avoid rounding effects at small sizes
z = (lv - mu) / sd
print('KS of log size (v>=3, truncated) — skip; moments of log v:', round(np.log(v).mean(), 4), round(np.log(v).std(), 4))
qs = [0.5, 0.9, 0.99, 0.999, 0.9999, 0.99999]
print('quantiles obs vs model', [(q, np.quantile(v, q), round(np.exp(mu + sd * norm.ppf(q)), 1)) for q in qs])
print('log size by regime', m.groupby('hi').vol.apply(lambda x: (np.log(x).mean().round(4), np.log(x).std().round(4), len(x))).to_dict())
b = pd.cut(m.rel, [-5, 0, 3, 3.5, 6, 20, np.inf], right=False)
print('maker size/side by time since shock onset'); print(m.groupby(b, observed=True).agg(n=('vol', 'size'), lmean=('vol', lambda x: np.log(x).mean()), lsd=('vol', lambda x: np.log(x).std()), buy=('side', lambda s: (s == 0).mean())).round(4).to_string())
# side choice vs imbalance & recent trend (calm, outside shocks)
M = m[(m.rel == np.inf)].copy()
M['imb'] = M.bid_qty_before / (M.bid_qty_before + M.ask_qty_before)
def mid_at(x): return mid[np.clip(np.searchsorted(tt, x, 'right') - 1, 0, None)]
M['tr1'] = M[['bid_before', 'ask_before']].mean(1) - mid_at(M.t.to_numpy() - 1)
M['tr10'] = M[['bid_before', 'ask_before']].mean(1) - mid_at(M.t.to_numpy() - 10)
for c in ['imb', 'tr1', 'tr10']:
    q = pd.qcut(M[c], 10, duplicates='drop')
    print(f'P(buy) by {c} decile:', M.groupby(q, observed=True).side.apply(lambda s: round((s == 0).mean(), 4)).tolist())
# placement relative to own-side touch / opposite touch by regime
M['dist_opp'] = np.where(M.side == 0, M.ask_before - M.px, M.px - M.bid_before)
print('distance to opposite touch at insert (ticks) by regime (share):')
print(pd.crosstab(M.hi, M.dist_opp.clip(0, 8), normalize='index').round(3).to_string())
print('mean spread by regime', M.groupby('hi').apply(lambda x: (x.ask_before - x.bid_before).mean()).round(3).to_dict())
