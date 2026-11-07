"""Event-aligned profiles around shock onsets (maker rates, taker rates/sides/sizes, mid move, cancels)."""
import numpy as np, pandas as pd
from pathlib import Path
D = Path(__file__).parent; C = D / 'cache'
r = pd.read_pickle(C / 'rows.pkl'); ev = pd.read_csv(D / 'shock_events.csv')
ins = r[r.op == 0]; can = r[r.op == 1]
mid = ((r.bid_after + r.ask_after) / 2).to_numpy(); tt = r.t.to_numpy()
def mid_at(x): return mid[np.clip(np.searchsorted(tt, x, 'right') - 1, 0, None)]
edges = np.arange(-5, 20.01, 0.5)
rows = []
for o in ev.onset:
    rel_i = ins.t.to_numpy() - o
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (rel_i >= lo) & (rel_i < hi)
        x = ins[s]
        m = x[~x.ioc]; k = x[x.ioc]
        rows.append(dict(o=o, lo=lo, mb=(m.side == 0).sum(), ma=(m.side == 1).sum(), tb=(k.side == 0).sum(), ts=(k.side == 1).sum(),
                         tvol=k.vol.sum(), mvol=m.vol.sum(),
                         canc=((can.t - o >= lo) & (can.t - o < hi)).sum()))
P = pd.DataFrame(rows)
g = P.groupby('lo').mean() / 0.5
g['tk_size'] = P.groupby('lo').tvol.sum() / (P.groupby('lo').tb.sum() + P.groupby('lo').ts.sum())
g['mk_size'] = P.groupby('lo').mvol.sum() / (P.groupby('lo').mb.sum() + P.groupby('lo').ma.sum())
pd.set_option('display.width', 200)
print(g.drop(columns=['o', 'tvol', 'mvol']).round(2).to_string())
# per-event: price move and taker side imbalance in burst [3, 10)
out = []
for o in ev.onset:
    k = ins[ins.ioc & (ins.t >= o + 3) & (ins.t < o + 10)]
    k0 = ins[ins.ioc & (ins.t >= o) & (ins.t < o + 3)]
    sgn = np.where(k.side == 0, 1, -1)
    out.append(dict(onset=o, mid_m5=mid_at(o - 5), mid0=mid_at(o), mid3=mid_at(o + 3), mid10=mid_at(o + 10), mid30=mid_at(o + 30), mid60=mid_at(o + 60),
                    n_burst=len(k), buy_frac=(k.side == 0).mean(), signed_vol=(sgn * k.vol).sum(), n_drought=len(k0), drought_buy=(k0.side == 0).mean()))
E = pd.DataFrame(out)
for a, b in [('mid_m5', 'mid0'), ('mid0', 'mid3'), ('mid3', 'mid10'), ('mid10', 'mid30'), ('mid30', 'mid60')]:
    E[f'd_{a}_{b}'] = E[b] - E[a]
print(E.round(2).to_string())
print('corr burst signed vol vs moves:')
print(E[[c for c in E.columns if c.startswith('d_')] + ['signed_vol', 'buy_frac']].corr().round(2)['signed_vol'].to_string())
E.to_csv(D / 'shock_event_moves.csv', index=False)
