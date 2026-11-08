"""Informedness with exact S: E[sg*(S(t+h) - S_after)] for takers, by regime, by fill status, by size.
S_after = s_hat at the next insert after the taker (impact already applied). Under coin-flip takers with transient
impact, this is <= 0 (own-impact decay); positive = information/momentum."""
import numpy as np, pandas as pd
from pathlib import Path
D = Path(__file__).parent; C = D / 'cache'
r = pd.read_pickle(C / 'rows.pkl'); ins = r[r.op == 0].reset_index(drop=True)
S = pd.read_parquet(D.parent / 's_dynamics/s_path.parquet')
assert len(S) == len(ins)
t = ins.t.to_numpy(); s = S.s_hat.to_numpy(); vol = S.vol.to_numpy(); shock = S.shock.to_numpy()
o = pd.read_pickle(C / 'orders.pkl')
ioc = ins.ioc.to_numpy(); idx = np.flatnonzero(ioc)
filled = o.filled.to_numpy()[idx] if len(o) == len(ins) else None
sg = np.where(ins.side.to_numpy()[idx] == 0, 1, -1)
def S_at(x): return s[np.clip(np.searchsorted(t, x, 'right') - 1, 0, None)]
s_before = s[idx - 1]; s_after = s[np.minimum(idx + 1, len(s) - 1)]
ok = ~shock[idx] & (idx > 0)
print('takers', len(idx), 'ok', ok.sum(), 'filled available', filled is not None)
imp = sg * (s_after - s_before)
for v in [False, True]:
    m = ok & (vol[idx] == v)
    line = f'vol={v}: own jump E[sg*dS] {imp[m].mean():+.4f}; '
    for h in [0.1, 0.5, 1, 5, 15, 30, 60]:
        d = sg * (S_at(t[idx] + h) - s_after); line += f'h{h}: {d[m].mean():+.3f}±{d[m].std() / np.sqrt(m.sum()):.3f} '
    print(line)
    if filled is not None:
        for nm, f in [('unfilled', filled == 0), ('filled', filled > 0)]:
            mm = m & f; line = f'   {nm} n={mm.sum()} own {imp[mm].mean():+.4f} '
            for h in [1, 5, 30]:
                d = sg * (S_at(t[idx] + h) - s_after); line += f'h{h}: {d[mm].mean():+.3f}±{d[mm].std() / np.sqrt(mm.sum()):.3f} '
            print(line)
# past S move vs sign (momentum / contrarian takers)
for h in [0.5, 1, 5, 30]:
    past = s_before - S_at(t[idx] - h)
    for v in [False, True]:
        m = ok & (vol[idx] == v); print(f'corr(sign, past {h}s S move) vol={v}: {np.corrcoef(sg[m], past[m])[0, 1]:+.4f} (n {m.sum()}, se {1 / np.sqrt(m.sum()):.4f})')
# sign vs S - mid (does the taker side depend on where S is in the spread?)
mid = ((ins.bid_before + ins.ask_before) / 2).to_numpy()[idx]
x = s_before - mid; m = ok & np.isfinite(x)
print('corr(sign, S_before - mid_before):', np.corrcoef(sg[m], x[m])[0, 1].round(4))
print(pd.Series(sg[m]).groupby(pd.cut(x[m], [-5, -1, -0.5, -0.25, 0, 0.25, 0.5, 1, 5])).agg(['mean', 'count']).round(3).to_string())
