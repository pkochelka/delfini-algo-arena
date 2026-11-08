"""How many live ladder (chain) orders share one price level? Sample at burst starts, per side, levels -2..-7."""
import numpy as np, pandas as pd
from collections import Counter
o = pd.read_pickle('cache/chains.pkl')
g = np.load('cache/regime_sz.npz')['post']
o['vol'] = g[(o.t_us//1_000_000).to_numpy(), 1] > .5
lad = o[(o.L >= 2) & (o.L <= 6) & ~o.ioc.astype(bool)].copy()
lad['end_t'] = lad.end_t_us.fillna(6e9)
t = lad.t_us.to_numpy(); e = lad.end_t.to_numpy(); s = lad.sg.to_numpy(); pr = lad.price.to_numpy()
starts = lad[lad.k == 0]
cnt = {True: Counter(), False: Counter()}; tot = {True: Counter(), False: Counter()}
for i, r in enumerate(starts.itertuples()):
    if i % 3: continue
    j0 = np.searchsorted(t, r.t_us - 120e6); j1 = np.searchsorted(t, r.t_us)
    live = (e[j0:j1] > r.t_us) & (s[j0:j1] == r.sg)
    c = Counter(pr[j0:j1][live])
    for p, n in c.items(): cnt[r.vol][n] += 1
    tot[r.vol][live.sum()] += 1
for v in (False, True):
    print('VOL' if v else 'CALM', 'orders per occupied level:', sorted(cnt[v].items()))
    print('   live ladder orders per side:', sorted(tot[v].items())[:40])
