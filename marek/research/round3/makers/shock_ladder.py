"""At each shock jump (flow t_jump, jump J): resting maker orders on the stale side (asks if J>0, bids if J<0) priced
beyond S_after by > 0.5 tick. Split ladder (chain) vs other. Their survival after the jump and fill share."""
import numpy as np, pandas as pd
o = pd.read_pickle('cache/chains.pkl')
S = pd.read_parquet('../s_dynamics/s_path.parquet'); o['S'] = S.s_hat.to_numpy()
ev = pd.read_csv('../flow/shock_events.csv')
m = o[~o.ioc.astype(bool)].copy(); m['end_t'] = m.end_t_us.fillna(6e9)
m['lad'] = (m.L >= 2) & (m.L <= 6)
# ladder-like singletons: size band by regime (rough): calm 90-320; we just report chains and "deep" others
St = o.t_us.to_numpy(); Sv = o.S.to_numpy()
rows = []; surv = {True: [], False: []}
grid = np.array([5, 10, 20, 50, 100, 200, 500, 1000, 2000]) * 1000
for _, e in ev.iterrows():
    tj = int(e.t_jump * 1e6); J = e.jump; side = -1 if J > 0 else 1     # stale side: asks (sg=-1) when J>0
    Sa = Sv[np.searchsorted(St, tj + 300_000)]       # S 300 ms after jump
    live = m[(m.t_us < tj) & (m.end_t > tj) & (m.sg == side)]
    edge = (Sa - live.price) * (-side)    # for asks (side -1): S_after - price > 0 means ask below S
    st = live[edge > 0.5]
    for lad in (True, False):
        s = st[st.lad == lad]
        q = s.volume.sum()
        rows.append(dict(t=e.t_jump, J=J, lad=lad, n=len(s), qty=q, edge_w=(edge[s.index]*s.volume).sum()/max(q, 1),
                         filled_share=(s.filled.sum()/max(q, 1)), mean_life_after_ms=((np.minimum(s.end_t, 6e9) - tj)/1e3).mean() if len(s) else np.nan))
        if q: surv[lad].append([(s.volume * (s.end_t > tj + d)).sum() for d in grid] + [q])
r = pd.DataFrame(rows)
pd.set_option('display.width', 200)
print(r.pivot_table(index=['t', 'J'], columns='lad', values=['n', 'qty', 'edge_w', 'filled_share', 'mean_life_after_ms']).round(2).to_string())
for lad in (True, False):
    a = np.array(surv[lad]); tot = a[:, -1].sum()
    print('ladder' if lad else 'other', 'stale qty total', tot, 'share still resting after (ms)', dict(zip((grid/1000).astype(int), (a[:, :-1].sum(0)/tot).round(3))))
