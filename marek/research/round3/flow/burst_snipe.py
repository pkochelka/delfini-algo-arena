"""First 0.5 s after each shock jump: per-taker timing, side vs jump, size, limit vs new S, fill. Fine profile."""
import numpy as np, pandas as pd
from pathlib import Path
D = Path(__file__).parent; C = D / 'cache'
X = pd.read_csv(D / 'shock_events.csv'); r = pd.read_pickle(C / 'rows.pkl'); ins = r[r.op == 0].reset_index(drop=True)
S = pd.read_parquet(D.parent / 's_dynamics/s_path.parquet'); o = pd.read_pickle(C / 'orders.pkl')
ti = ins.t.to_numpy(); s = S.s_hat.to_numpy(); ioc = ins.ioc.to_numpy()
idx = np.flatnonzero(ioc); tk = ti[idx]; sg = np.where(ins.side.to_numpy()[idx] == 0, 1, -1)
rows = []
for _, e in X.iterrows():
    m = np.flatnonzero((tk > e.t_jump - 0.0005) & (tk < e.t_jump + 0.5))
    for j in m:
        i = idx[j]
        rows.append(dict(shock=e.onset, dt_ms=(tk[j] - e.t_jump) * 1e3, dirn=sg[j] * np.sign(e.jump), J=abs(e.jump), vol=ins.vol[i],
                         x_new=sg[j] * (ins.px[i] - s[i - 1]), filled=o.filled[i], crossed=o.crossed[i] if 'crossed' in o else np.nan))
B = pd.DataFrame(rows)
e = [0, 10, 25, 50, 75, 100, 150, 200, 250, 350, 500]
print(B.groupby(pd.cut(B.dt_ms, e)).agg(n=('dirn', 'size'), share_dir=('dirn', lambda x: (x > 0).mean()), med_size=('vol', 'median'),
      x_med=('x_new', 'median'), fill_frac=('filled', lambda x: (x > 0).mean())).round(3).to_string())
W = B[B.dt_ms < 250]
for d in [1, -1]:
    Z = W[W.dirn == d]; print(f'dir {d}: n {len(Z)} x_new quantiles {np.round(np.quantile(Z.x_new, [.1, .25, .5, .75, .9]), 2)} logsize mean {np.log(Z.vol).mean():.2f} filled share {(Z.filled > 0).mean():.2f}')
# per shock count of jump-direction takers in first 250 ms vs |J|
g = W.groupby('shock').agg(nd=('dirn', lambda x: (x > 0).sum()), no=('dirn', lambda x: (x < 0).sum()), J=('J', 'first'))
print(g.sort_values('J').T.to_string())
print('corr(nd - no, |J|)', np.corrcoef(g.nd - g.no, g.J)[0, 1].round(2))
