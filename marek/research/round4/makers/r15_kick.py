"""Does S move at the agent's fills (step) or continuously (drift)? Side-signed S change in windows around agent fills
vs around other times inside chains; and a regression of the per-chain S slope on time vs number of fills."""
import numpy as np, pandas as pd
S = pd.read_parquet('../s_dynamics/s_path.parquet')
st = S.t_us.to_numpy(); sh = S.s_hat.to_numpy()
def s(t): return sh[np.clip(np.searchsorted(st, t, 'right') - 1, 0, len(st) - 1)]
a = pd.read_pickle('cache/r15_v2.pkl')
o = pd.read_pickle('cache/orders.pkl')
fills = a[a.end == 'filled']
tf = fills.end_t_us.to_numpy().astype(np.int64); sgf = fills.sg.to_numpy()
# the taker that filled: its volume/filled and S kick expectation
tk = o[o.ioc].set_index('t_us')
for w in [20e3, 100e3, 300e3]:
    d = sgf * (s(tf + w) - s(tf - w))
    print(f'agent fill, S change over ±{w/1e3:.0f} ms (agent-signed): mean {d.mean():.3f} med {np.median(d):.3f} sd {d.std():.3f}')
# random times inside chains
rng = np.random.default_rng(0)
g = a.groupby('chain').agg(t0=('t_us', 'min'), t1=('end_t_us', 'max'), sg=('sg', 'first'))
rt, rs = [], []
for r in g.itertuples():
    n = 30
    rt.append(rng.uniform(r.t0, r.t1, n)); rs.append(np.full(n, r.sg))
rt = np.concatenate(rt).astype(np.int64); rs = np.concatenate(rs)
for w in [20e3, 100e3, 300e3]:
    d = rs * (s(rt + w) - s(rt - w))
    print(f'random in-chain time, S change over ±{w/1e3:.0f} ms: mean {d.mean():.3f} med {np.median(d):.3f}')
# the takers that hit the agent: side and volume, filled
tt = tk.loc[tk.index.intersection(tf)]
print('takers filling agent: n', len(tt), 'side opposite to agent share', None, 'vol med', tt.volume.median(), 'filled med', tt.filled.median())
