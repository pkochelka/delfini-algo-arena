"""15-lot agent chains vs S (s_dynamics v0): S moves around chains signed by agent side, regime, S vs agent price."""
import numpy as np, pandas as pd
S = pd.read_parquet('../s_dynamics/s_path.parquet')
st = S.t_us.to_numpy(); sh = S.s_hat.to_numpy(); vol = S.vol.to_numpy()
def s(t): return sh[np.clip(np.searchsorted(st, t, 'right') - 1, 0, len(st) - 1)]
def v(t): return vol[np.clip(np.searchsorted(st, t, 'right') - 1, 0, len(st) - 1)]
a = pd.read_pickle('cache/r15_v2.pkl')
g = a.groupby('chain').agg(t0=('t_us', 'min'), t1=('end_t_us', 'max'), sg=('sg', 'first'), filled=('filled', 'sum')).reset_index()
tw = pd.read_pickle('cache/tw.pkl'); tw = tw[tw.tw_L >= 2]
twg = tw.groupby('tw_chain').agg(t0=('t_us', 'min'), t1=('t_us', 'max'), sg=('sg', 'first')).reset_index()
def table(g, name):
    rows = {}
    for lab, a0, a1 in [('-30..-10', -30, -10), ('-10..-2', -10, -2), ('-2..0', -2, 0)]:
        rows[lab] = g.sg * (s(g.t0 + a1 * 1e6) - s(g.t0 + a0 * 1e6))
    rows['0..+2'] = g.sg * (s(g.t0 + 2e6) - s(g.t0))
    rows['during'] = g.sg * (s(g.t1) - s(g.t0))
    for h in [2, 10, 30, 60]:
        rows[f'end..+{h}'] = g.sg * (s(g.t1 + h * 1e6) - s(g.t1))
    r = pd.DataFrame(rows)
    print(f'== {name}: n={len(g)}; side-signed S change (ticks), vol share at t0 {v(g.t0).mean():.2f}')
    print(pd.DataFrame({'mean': r.mean(), 'median': r.median(), 'sd': r.std(), 't': r.mean() / r.std() * np.sqrt(len(r))}).round(2).T.to_string())
table(g, '15-lot agent chains (sg = agent order side)')
table(twg, 'TWAP (sg = taker side)')
# agent order price vs S at insert
ai = a.index.to_numpy()
d = (S.s_hat.to_numpy()[ai] - a.price.to_numpy()) * a.sg.to_numpy()  # + = S beyond the agent's price in its favour? for bid: S - bid
print('S - price (signed: bid S-p, ask p-S) at agent insert quantiles', np.quantile(d, [.05, .25, .5, .75, .95]).round(2))
print('vol share of agent orders', S.vol.to_numpy()[ai].mean(), ' overall', S.vol.mean())
