"""Tradeability: time from episode start to the first observable agent tell (first 1 us re-post), and the side-signed
S / mid move from that moment to +1, +2, +5 s and to the episode end."""
import numpy as np, pandas as pd
S = pd.read_parquet('../s_dynamics/s_path.parquet')
st = S.t_us.to_numpy(); sh = S.s_hat.to_numpy()
q = pd.read_pickle('cache/quotes.pkl'); qt = q.t_us.to_numpy(); mid = ((q.bid + q.ask) / 2).ffill().to_numpy()
def s(t): return sh[np.clip(np.searchsorted(st, t, 'right') - 1, 0, len(st) - 1)]
def m(t): return mid[np.clip(np.searchsorted(qt, t, 'right') - 1, 0, len(qt) - 1)]
a = pd.read_pickle('cache/r15_v2.pkl')
e = pd.read_csv('r15_episodes.csv')
rows = []
for r in e.itertuples():
    x = a[(a.t_us >= r.t0) & (a.t_us <= r.t1) & (a.sg == r.sg) & (a.k >= 1)].sort_values('t_us')
    td = x.t_us.iloc[0]
    d = dict(det_s=(td - r.t0) / 1e6, left_s=(r.t1 - td) / 1e6)
    for h in [1, 2, 5]:
        d[f'S+{h}'] = r.sg * (s(td + h * 1e6) - s(td)); d[f'mid+{h}'] = r.sg * (m(td + h * 1e6) - m(td))
    d['S_to_end'] = r.sg * (s(r.t1) - s(td)); d['mid_to_end'] = r.sg * (m(r.t1) - m(td))
    rows.append(d)
d = pd.DataFrame(rows)
print(d.describe().round(2).T[['mean', '50%', 'std', 'min', 'max']].to_string())
print('t-stats', (d.mean() / d.std() * np.sqrt(len(d))).round(1).to_dict())
# every re-post as a signal: mid move over next 1/2/5 s from each agent link (k>=1) insert
x = a[a.k >= 1]
for h in [0.5, 1, 2, 5]:
    v = x.sg * (m(x.t_us + h * 1e6) - m(x.t_us))
    print(f'from each agent re-post: side-signed mid move +{h}s mean {v.mean():.2f} med {v.median():.2f} n {len(v)}')
