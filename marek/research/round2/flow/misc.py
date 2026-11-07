"""Seasonality / structural breaks / lifetimes by regime / maker gap distribution."""
import numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
D = Path(__file__).parent; C = D / 'cache'
z = np.load(C / 'bins1s.npz'); mask = z['mask']; hi = np.load(C / 'hmm_taker_gamma2.npy')[:, 1] > 0.5
print('per 1000 s block: maker rate, calm taker rate/side, volatile taker rate/side, volatile share')
for b in range(6):
    s = slice(b * 1000, (b + 1) * 1000); m = mask[s]; h = hi[s]
    print(f'  {b*1000:4d}-{(b+1)*1000:4d}: maker {z["mk"][s][m].mean():6.2f}  calm taker {z["tk"][s][m & ~h].mean()/2:5.3f}  vol taker {np.nanmean(z["tk"][s][m & h])/2 if (m&h).any() else float("nan"):5.3f}  vol share {h.mean():.3f}')
# periodicity: periodogram of maker and taker counts (masked filled with mean)
for name in ['mk', 'tk']:
    x = np.where(mask, z[name], z[name][mask].mean()); x = x - x.mean()
    P = np.abs(np.fft.rfft(x)) ** 2; f = np.fft.rfftfreq(len(x), 1.0)
    top = np.argsort(P[1:])[::-1][:6] + 1
    print(f'{name} top periodogram periods (s):', np.round(1 / f[top], 1), 'power ratio to median', np.round(P[top] / np.median(P[1:]), 1))
o = pd.read_pickle(C / 'orders.pkl'); r = pd.read_pickle(C / 'rows.pkl')
g = o[~o.ioc.astype(bool) & (o.end_t_us > 0)].copy()
g['life'] = (g.end_t_us - g.t_us) / 1e6; g['sec'] = (g.t_us / 1e6).astype(int).clip(0, 5999)
g['hi'] = hi[g.sec]; g['ok'] = mask[g.sec]
ev = pd.read_csv(D / 'shock_events.csv'); rel = np.full(len(g), np.inf); ts = g.t_us.to_numpy() / 1e6
for on in ev.onset:
    d = ts - on; s = (d > -1) & (d < 10); rel[s] = d[s]
g['rel'] = rel
print('maker lifetime by regime: ', g[g.ok].groupby('hi').life.describe(percentiles=[.1, .5, .9]).round(3).to_string())
print('lifetime by time since shock onset:'); print(g[np.isfinite(g.rel)].groupby(pd.cut(g.rel[np.isfinite(g.rel)], [-1, 0, 1, 2, 2.5, 3, 3.5, 5, 10])).life.agg(['mean', 'median', 'count']).round(3).to_string())
print('end reasons', g.end.value_counts().to_dict())
mt = r.t[(r.op == 0) & ~r.ioc].to_numpy(); gap = np.diff(mt); okg = mask[mt[1:].astype(int).clip(0, 5999)]
gp = gap[okg]; print('maker gaps (calm/vol, no shock): mean %.6f cv %.4f min %.2e KS p %.3f' % (gp.mean(), gp.std() / gp.mean(), gp.min(), stats.kstest(gp, 'expon', (0, gp.mean())).pvalue))
