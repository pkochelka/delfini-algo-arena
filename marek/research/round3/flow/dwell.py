"""Regime dwell laws from s_dynamics vol_segments.csv (S-based switch times), cross-checked with flow's Viterbi
(cache/regimes_T0.1.csv, segments with LLR>100). Compare to round 2. -> regimes.csv (final table)."""
import numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
D = Path(__file__).parent; C = D / 'cache'
V = pd.read_csv(D.parent / 's_dynamics/vol_segments.csv')
ev = pd.read_csv(D / 'shock_events.csv')
mine = pd.read_csv(C / 'regimes_T0.1.csv'); mv = mine[(mine.regime == 'volatile') & (mine.llr > 100)]
# match
for _, s in V.iterrows():
    j = np.argmin(np.abs(mv.start.to_numpy() - s.start)); m = mv.iloc[j]
    flag = '' if abs(m.start - s.start) < 1 and abs(m.end - s.end) < 1 else '  <-- differs'
    print(f'S-based {s.start:9.3f}-{s.end:9.3f} | flow {m.start:9.3f}-{m.end:9.3f}  d_start {m.start - s.start:+6.3f} d_end {m.end - s.end:+6.3f}{flag}')
print('flow segments not matched:', [(round(a, 2), round(b, 2)) for a, b in zip(mv.start, mv.end) if np.min(np.abs(V.start.to_numpy() - a)) > 2])
rows = []; prev = 0.0
for _, s in V.iterrows():
    rows.append(dict(start=prev, end=s.start, regime='calm')); rows.append(dict(start=s.start, end=s.end, regime='volatile')); prev = s.end
rows.append(dict(start=prev, end=6000.0, regime='calm'))
R = pd.DataFrame(rows); R['dur'] = R.end - R.start
R['censored'] = (R.start == 0) | (R.end == 6000)
R['shocks_inside'] = [((ev.onset > a) & (ev.onset < b)).sum() for a, b in zip(R.start, R.end)]
R.to_csv(D / 'regimes.csv', index=False)
for reg in ['calm', 'volatile']:
    d = R[(R.regime == reg) & ~R.censored].dur.to_numpy()
    T = R[R.regime == reg].dur.sum(); nexit = (~R[R.regime == reg].censored).sum() if reg == 'volatile' else len(V)
    mle = T / len(V)  # exits = number of switches out (28 each way, censoring at end)
    print(f'{reg}: n {len(d)} mean {d.mean():.1f} median {np.median(d):.1f} sd {d.std():.1f} min {d.min():.1f} max {d.max():.1f}; '
          f'total time {T:.0f}s occ {T / 6000:.3f}; MLE mean dwell (time/exits) {mle:.1f}; CV {d.std() / d.mean():.2f}')
    print('   KS exp(mean) p=%.3f; KS shifted exp (min + Exp) p=%.3f; sorted %s' % (stats.kstest(d, 'expon', (0, d.mean())).pvalue,
          stats.kstest(d, 'expon', (d.min(), d.mean() - d.min())).pvalue, np.round(np.sort(d), 1).tolist()))
# r2 comparison
print('r2: calm mean ≈ 247 s, volatile mean ≈ 43 s (7–229), occ 0.147')
# shocks vs switches: time from each switch to nearest shock onset / end
sw = np.r_[V.start, V.end]
dd = np.array([np.min(np.abs(ev.onset.to_numpy() + 3 - x)) for x in sw])
print('switch distance to nearest jump (s): min', np.round(np.sort(dd)[:8], 1).tolist(), ' expected #<10 s if independent ≈ %.1f' % (len(sw) * 27 * 20 / 6000))
print('regime at jump times:', ['V' if ((V.start < j) & (V.end > j)).any() else 'c' for j in ev.onset + 3])
