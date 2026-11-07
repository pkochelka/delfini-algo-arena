"""Do makers quote off S or off the touch? Among unclipped calm inserts, mean (S - price) for bids / (price - S) for
asks by the same-side touch's distance from S. Under the S rule it does not depend on the same-side touch."""
import numpy as np, pandas as pd
pd.set_option('display.width', 200)
from s_load import with_s
import states
o = with_s(); g = o[~o.ioc & o.bid.notna() & o.ask.notna()].copy()
g['st'] = states.state(g.t_us.to_numpy())
b = g.sg == 1
g['gap'] = np.where(b, g.S - g.price, g.price - g.S)
g['touch_gap'] = np.where(b, g.S - g.bid, g.ask - g.S)  # same-side touch distance from S
g['clipped'] = np.where(b, g.price == g.ask - 1, g.price == g.bid + 1)
for s_ in ['calm', 'vol']:
    x = g[(g.st == s_) & ~g.clipped]
    t = x.groupby(pd.cut(x.touch_gap, [-5, 0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 6]), observed=True).gap.agg(['mean', 'median', 'count'])
    print(s_, 'unclipped: gap S-price by same-side touch distance from S\n', t.round(3))
    print(s_, 'price relative to same-side touch (ticks, + = inside):',
          (np.where(x.sg == 1, x.price - x.bid, x.ask - x.price)).clip(-6, 3).astype(int).tolist().__len__())
    rel = pd.Series(np.where(x.sg == 1, x.price - x.bid, x.ask - x.price)).clip(-6, 3)
    print((rel.value_counts(normalize=True).sort_index()).round(3).to_dict())

# model replay: same S and book, price from the S rule with the fitted D law; compare the same table
rng = np.random.default_rng(0)
m = np.where(g.st == 'vol', 2.62, 1.0)
D = m * (0.0887 + rng.exponential(0.4201, len(g)))
S = g.S.to_numpy() + g.S_sd.to_numpy() * rng.standard_normal(len(g))
pb = np.minimum(np.floor(S - D), g.ask - 1); pa = np.maximum(np.ceil(S + D), g.bid + 1)
g['sim_price'] = np.where(g.sg == 1, pb, pa)
g['sim_gap'] = np.where(g.sg == 1, g.S - g.sim_price, g.sim_price - g.S)
g['sim_clipped'] = np.where(g.sg == 1, g.sim_price == g.ask - 1, g.sim_price == g.bid + 1)
for s_ in ['calm', 'vol']:
    x = g[g.st == s_]
    cut = pd.cut(x.touch_gap, [-5, 0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 6])
    t = pd.DataFrame({'obs_gap_unclipped': x[~x.clipped].groupby(cut[~x.clipped], observed=True).gap.mean(),
                      'sim_gap_unclipped': x[~x.sim_clipped].groupby(cut[~x.sim_clipped], observed=True).sim_gap.mean(),
                      'obs_clip_share': x.groupby(cut, observed=True).clipped.mean(),
                      'sim_clip_share': x.groupby(cut, observed=True).sim_clipped.mean()})
    print(s_, 'observed vs S-rule replay'); print(t.round(3).to_string())
