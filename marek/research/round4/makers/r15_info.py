"""Is the 15-lot agent informed? Mid moves before / during / after its chains, signed by the agent's side
(+ = price moved in the agent's favour: up for a buyer, down for a seller... we sign by agent side sg)."""
import numpy as np, pandas as pd
q = pd.read_pickle('cache/quotes.pkl')
qt = q.t_us.to_numpy(); mid = ((q.bid + q.ask) / 2).ffill().to_numpy()
def m(t): return mid[np.clip(np.searchsorted(qt, t, 'right') - 1, 0, len(qt) - 1)]
a = pd.read_pickle('cache/r15_v2.pkl')
g = a.groupby('chain').agg(t0=('t_us', 'min'), t1=('end_t_us', 'max'), sg=('sg', 'first'), filled=('filled', 'sum')).reset_index()
tw = pd.read_pickle('cache/tw.pkl'); tw = tw[tw.tw_L >= 2]
twg = tw.groupby('tw_chain').agg(t0=('t_us', 'min'), t1=('t_us', 'max'), sg=('sg', 'first')).reset_index()
def table(g, name):
    rows = {}
    for lab, a0, a1 in [('-60..-30', -60, -30), ('-30..0', -30, 0), ('-10..0', -10, 0)]:
        rows[lab] = g.sg * (m(g.t0 + a1 * 1e6) - m(g.t0 + a0 * 1e6))
    rows['during'] = g.sg * (m(g.t1) - m(g.t0))
    for h in [10, 30, 60, 120]:
        rows[f'end..+{h}'] = g.sg * (m(g.t1 + h * 1e6) - m(g.t1))
    r = pd.DataFrame(rows)
    print(f'== {name}: n={len(g)}; mean (sd) of side-signed mid change, ticks')
    print(pd.DataFrame({'mean': r.mean(), 'sd': r.std(), 't': r.mean() / r.std() * np.sqrt(len(r))}).round(2).T.to_string())
table(g, '15-lot agent chains (sg = agent order side, +1 bid)')
table(twg, 'TWAP episodes (sg = taker side)')
g['dur'] = (g.t1 - g.t0) / 1e6
print(g.assign(t0=g.t0 / 1e6).to_string())
