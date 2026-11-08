"""Band hypothesis: near-end bursts (start -2, end > -7) follow the touch moving away by ~L; far-end bursts (end -7, start < -2)
follow it moving toward by ~L. Crosstab L vs touch change over the previous W ms (signed: + = away = bid up/ask down? see below)."""
import numpy as np, pandas as pd
o = pd.read_pickle('cache/chains.pkl')
S = pd.read_parquet('../s_dynamics/s_path.parquet')
o['S'] = S.s_hat.to_numpy()
g = np.load('cache/regime_sz.npz')['post']
o['vol'] = g[(o.t_us//1_000_000).to_numpy(), 1] > .5
o['rel'] = np.where(o.sg>0, o.price - o.bid, o.ask - o.price)
q = pd.read_pickle('cache/quotes.pkl'); qt = q.t_us.to_numpy(); qb = q.bid.to_numpy().astype(float); qa = q.ask.to_numpy().astype(float)
St = o.t_us.to_numpy(); Sv = o.S.to_numpy()
def at(tarr, varr, t): return varr[np.maximum(np.searchsorted(tarr, t, 'right') - 1, 0)]
lad = o[(o.L >= 2) & (o.L <= 6) & ~o.ioc.astype(bool)]
b = lad.groupby('chain').agg(t=('t_us','first'), sg=('sg','first'), r0=('rel','first'), L=('L','first'), vol=('vol','first'), S=('S','first'), p0=('price','first'))
b['rend'] = b.r0 - (b.L - 1)
b['type'] = np.select([(b.r0 == -2) & (b.rend > -7), (b.r0 < -2) & (b.rend == -7), (b.r0 == -2) & (b.rend == -7)], ['near', 'far', 'full'], 'other')
print(b.type.value_counts())
for W in (20e3, 50e3, 100e3, 200e3, 500e3):
    t = b.t.to_numpy() - 1; sg = b.sg.to_numpy()
    tn = np.where(sg > 0, at(qt, qb, t), at(qt, qa, t)); tp = np.where(sg > 0, at(qt, qb, t - W), at(qt, qa, t - W))
    b[f'away{int(W/1e3)}'] = (tn - tp) * sg            # + = touch moved away from the ladder (bid up)
    b[f'Saway{int(W/1e3)}'] = (Sv[np.searchsorted(St, t) - 1] - at(St, Sv, t - W)) * sg
for ty in ('near', 'far'):
    s = b[(b.type == ty) & b.vol]
    print('===', ty, 'vol, mean touch move away over W (ms) by L; and S move')
    print(s.groupby('L')[[c for c in b.columns if 'away' in c]].mean().round(2))
s = b[(b.type == 'near') & b.vol]
print(pd.crosstab(s.L, s.away200.clip(-3, 6)))
s = b[(b.type == 'far') & b.vol]
print(pd.crosstab(s.L, s.away200.clip(-6, 3)))
b.to_pickle('cache/bursts.pkl')
