"""Merge 15-lot agent instances into episodes (same side, gap < 3 s) and profile S around episode start / end."""
import numpy as np, pandas as pd
S = pd.read_parquet('../s_dynamics/s_path.parquet')
st = S.t_us.to_numpy(); sh = S.s_hat.to_numpy()
def s(t): return sh[np.clip(np.searchsorted(st, t, 'right') - 1, 0, len(st) - 1)]
a = pd.read_pickle('cache/r15_v2.pkl')
g = a.groupby('chain').agg(t0=('t_us', 'min'), t1=('end_t_us', 'max'), sg=('sg', 'first'), filled=('filled', 'sum'),
                           end=('end', 'last')).sort_values('t0').reset_index()
ep = []; cur = None
for r in g.itertuples():
    if cur is not None and r.sg == cur['sg'] and r.t0 < cur['t1'] + 3e6:
        cur['t1'] = max(cur['t1'], r.t1); cur['n'] += 1; cur['filled'] += r.filled; cur['ends'].append(r.end)
    else:
        if cur: ep.append(cur)
        cur = dict(t0=r.t0, t1=r.t1, sg=r.sg, n=1, filled=r.filled, ends=[r.end])
ep.append(cur)
e = pd.DataFrame(ep)
e['dur'] = (e.t1 - e.t0) / 1e6
e['dS'] = e.sg * (s(e.t1) - s(e.t0))
e['lastend'] = e.ends.str[-1]
print(e.assign(t0=e.t0 / 1e6, t1=e.t1 / 1e6).drop(columns='ends').round(2).to_string())
print('episodes', len(e), 'instances per episode', e.n.value_counts().to_dict(), 'buy/sell', (e.sg > 0).sum(), (e.sg < 0).sum())
print('dur quantiles', e.dur.quantile([.1, .25, .5, .75, .9]).round(1).to_dict(), 'total', e.dur.sum().round(0))
gaps = np.diff(e.t0.values) / 1e6
print('start gaps quantiles', np.quantile(gaps, [.1, .25, .5, .75, .9]).round(0), 'mean', gaps.mean().round(0))
taus = np.array([-5, -2, -1, -0.5, -0.2, 0, 0.2, 0.5, 1, 2, 3, 5, 8, 10, 15, 20, 30])
for ref in ['t0', 't1']:
    P = np.array([e.sg * (s(e[ref] + x * 1e6) - s(e[ref])) for x in taus])
    print(f'episode aligned at {ref}: tau', taus.tolist()); print('  mean', P.mean(1).round(2).tolist()); print('  med ', np.median(P, 1).round(2).tolist())
e.drop(columns='ends').to_csv('r15_episodes.csv', index=False)
