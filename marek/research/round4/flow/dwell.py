"""Regime segments from s_dynamics/vol_segments.csv -> regimes.csv (start,end,regime,dur,censored,shocks_inside);
dwell laws (exponential KS, mean CI) and agreement with the flow 1 s HMM."""
import numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
D = Path(__file__).parent; C = D / 'cache'
vs = pd.read_csv(D.parent / 's_dynamics/vol_segments.csv').sort_values('start'); ev = pd.read_csv(D / 'shock_events.csv')
rows = []; t = 0.0
for a, b in zip(vs.start, vs.end):
    if a > t: rows.append((t, a, 'calm'))
    rows.append((a, b, 'volatile')); t = b
if t < 6000: rows.append((t, 6000.0, 'calm'))
R = pd.DataFrame(rows, columns=['start', 'end', 'regime']); R['dur'] = R.end - R.start
R['censored'] = (R.start <= 0.5) | (R.end >= 5999.5)
R['shocks_inside'] = [int(((ev.onset + 3 >= a) & (ev.onset + 3 < b)).sum()) for a, b in zip(R.start, R.end)]
R.round(4).to_csv(D / 'regimes.csv', index=False)
for g in ['calm', 'volatile']:
    d = R[(R.regime == g) & ~R.censored].dur.to_numpy(); n = len(d); mu = d.mean()
    lo, hi = 2 * n * mu / stats.chi2.ppf(.975, 2 * n), 2 * n * mu / stats.chi2.ppf(.025, 2 * n)
    print(f'{g}: n {n} mean {mu:.1f} (95% CI {lo:.0f}-{hi:.0f}) min {d.min():.1f} max {d.max():.1f} CV {d.std() / mu:.2f} KS exp p {stats.kstest(d / mu, "expon").pvalue:.2f}')
print('vol occupancy', (R[R.regime == 'volatile'].dur.sum() / 6000).round(3))
x = np.load(C / 'vol1s.npy'); lab = np.zeros(6000, int)
for a, b in zip(vs.start, vs.end): lab[int(np.ceil(a)):int(b)] = 1
print('agreement with flow 1 s HMM (smoothed):', (x == lab).mean().round(4))
# frac part of volatile starts (s_dynamics noticed integer-second starts)
fr = (vs.start + 0.5) % 1 - 0.5; print('vol start distance to integer s:', np.round(np.sort(np.abs(fr)), 3).tolist())
fe = (vs.end + 0.5) % 1 - 0.5; print('vol end   distance to integer s:', np.round(np.sort(np.abs(fe)), 3).tolist())
