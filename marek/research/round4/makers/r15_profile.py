"""Mean side-signed S path around 15-lot agent chain starts and ends (and the same for TWAP episodes)."""
import numpy as np, pandas as pd
S = pd.read_parquet('../s_dynamics/s_path.parquet')
st = S.t_us.to_numpy(); sh = S.s_hat.to_numpy()
def s(t): return sh[np.clip(np.searchsorted(st, t, 'right') - 1, 0, len(st) - 1)]
a = pd.read_pickle('cache/r15_v2.pkl')
g = a.groupby('chain').agg(t0=('t_us', 'min'), t1=('end_t_us', 'max'), sg=('sg', 'first'), filled=('filled', 'sum')).reset_index()
taus = np.array([-10, -5, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 30])
for ref in ['t0', 't1']:
    P = np.array([g.sg * (s(g[ref] + x * 1e6) - s(g[ref])) for x in taus])
    print(f'aligned at {ref}: mean / median / sd')
    print(pd.DataFrame({'tau': taus, 'mean': P.mean(1).round(2), 'med': np.median(P, 1).round(2), 'sd': P.std(1).round(2)}).T.to_string(header=False))
# slope during chain per chain: S(t1)-S(t0) / dur, plus linear-fit residual
g['dur'] = (g.t1 - g.t0) / 1e6
g['dS'] = g.sg * (s(g.t1) - s(g.t0))
g['rate'] = g.dS / g.dur
print(g[['t0', 'dur', 'filled', 'dS', 'rate']].assign(t0=g.t0 / 1e6).round(2).to_string())
print('rate quantiles', g.rate.quantile([.1, .25, .5, .75, .9]).round(2).to_dict(), 'corr(dS,dur)', np.corrcoef(g.dS, g.dur)[0, 1].round(2),
      'corr(dS, filled)', np.corrcoef(g.dS, g.filled)[0, 1].round(2))
