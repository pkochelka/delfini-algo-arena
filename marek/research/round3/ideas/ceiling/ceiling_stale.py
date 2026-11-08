"""Ceiling angle, step 2: stale-quote (IOC) ceiling per regime from _cache/ceil_stale.parquet."""
import sys, pathlib
import numpy as np, pandas as pd
H = pathlib.Path(__file__).parent
sys.path.insert(0, str(H.parents[2] / 'strategy'))
import eval3
on, reg = eval3.load_ctx('data/round3.csv')
st = pd.read_parquet(H / '_cache/ceil_stale.parquet')
tk = pd.read_parquet(H / '_cache/ceil_takers.parquet')
tr = pd.read_parquet(H.parents[2] / 'novelty/cache/round3_trades.parquet')
print('taker filled replay', tk.filled.sum(), 'novelty trades', tr.qty.sum())
st['reg'] = [eval3.label(t * 1e-6, on, reg) for t in st.t_us]
secs = {'calm': 0, 'volatile': 0, 'shock': 0}
for x in np.arange(0, 6000, 0.1):
    secs[eval3.label(x, on, reg)] += 0.1
print('secs', {k: round(v) for k, v in secs.items()})
st['v_arr'] = st.rem_arr * st.edge_arr
st['v_t'] = st.q * st.edge
print('\nall stale orders (edge>0.25 at first sight): value at first sight vs value reachable 5 ms later')
for thr in [0.25, 0.5, 1.0, 2.0, 4.0]:
    m = st.edge_arr > thr
    g = st[m].groupby('reg').agg(n=('oid', 'size'), lots=('rem_arr', 'sum'), val=('v_arr', 'sum'))
    print(f'edge_arr>{thr}:', {r: (int(g.loc[r, 'n']), int(g.loc[r, 'lots']), round(g.loc[r, 'val'])) for r in g.index})
g = st.groupby('reg').agg(lots_t=('q', 'sum'), val_t=('v_t', 'sum'), lots_a=('rem_arr', 'sum'))
print('at first sight (all):', g.round().to_dict())
print('share of first-sight lots still there at +5 ms:', (st.groupby('reg').rem_arr.sum() / st.groupby('reg').q.sum()).round(3).to_dict())
# opportunity events: orders first stale at the same t, same side
ok = st[(st.rem_arr > 0) & (st.edge_arr > 0.5)].copy()
ev = []
for (t, side), g in ok.groupby(['t_us', 'side']):
    g = g.sort_values('edge_arr', ascending=False)
    q = g.rem_arr.values; e = g.edge_arr.values
    cq = np.cumsum(q)
    def val(cap):
        take = np.clip(cap - (cq - q), 0, q)
        return float((take * e).sum())
    ev.append((t, side, g.reg.iloc[0], q.sum(), val(1e9), val(100), val(200), val(1000), e.max()))
ev = pd.DataFrame(ev, columns=['t_us', 'side', 'reg', 'Q', 'V', 'V100', 'V200', 'V1000', 'emax'])
ev.to_parquet(H / '_cache/ceil_events.parquet')
print('\nopportunity events (edge_arr>0.5, reachable): per regime')
g = ev.groupby('reg').agg(n=('Q', 'size'), Q=('Q', 'sum'), V=('V', 'sum'), V100=('V100', 'sum'), V200=('V200', 'sum'), V1000=('V1000', 'sum'))
g['per_s_V100'] = g.V100 / pd.Series(secs)
g['per_s_V200'] = g.V200 / pd.Series(secs)
print(g.round(0))
print('events with Q>100 per regime:', ev[ev.Q > 100].groupby('reg').size().to_dict(),
      ' value beyond 100 lots:', (ev.V - ev.V100).groupby(ev.reg).sum().round().to_dict())
# size distribution of events in volatile
for r in ['calm', 'volatile', 'shock']:
    e = ev[ev.reg == r]
    print(r, 'Q quantiles', np.percentile(e.Q, [50, 90, 99]).round(), 'emax quantiles', np.percentile(e.emax, [50, 90, 99]).round(2),
          'V by emax bin', e.groupby(pd.cut(e.emax, [0, 1, 2, 4, 10, 200])).V100.sum().round().to_dict())
# per-shock value
sh = ev[ev.reg == 'shock'].copy()
import bisect
sh['k'] = [bisect.bisect_right(on, t * 1e-6 + 1.0) - 1 for t in sh.t_us]
sh['rel'] = [t * 1e-6 - on[k] for t, k in zip(sh.t_us, sh.k)]
sj = sh[(sh.rel > 2.5) & (sh.rel < 3.6)]
print('\nshock jump windows (onset+2.5..3.6 s):', 'V', round(sj.V.sum()), 'V100', round(sj.V100.sum()), 'V1000', round(sj.V1000.sum()))
print(sj.groupby('k').agg(V=('V', 'sum'), V100=('V100', 'sum'), V1000=('V1000', 'sum'), Q=('Q', 'sum')).round().sort_values('V', ascending=False).head(10))
rest = sh[~((sh.rel > 2.5) & (sh.rel < 3.6))]
print('shock windows outside jump:', 'V', round(rest.V.sum()), 'V100', round(rest.V100.sum()), 'V200', round(rest.V200.sum()))
