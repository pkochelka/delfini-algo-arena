"""Shock events: gap distribution, burst profile (0.25 s), burst count vs |jump|, book depth in drought, coincidence with volatile regime."""
import numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
D = Path(__file__).parent; C = D / 'cache'
r = pd.read_pickle(C / 'rows.pkl'); ev = pd.read_csv(D / 'shock_events.csv'); E = pd.read_csv(D / 'shock_event_moves.csv')
hi = np.load(C / 'hmm_taker_gamma2.npy')[:, 1] > 0.5
tt = r.t.to_numpy(); mid = ((r.bid_after + r.ask_after) / 2).ffill().to_numpy()
def at(a, x): return a[np.clip(np.searchsorted(tt, x, 'right') - 1, 0, None)]
gaps = ev.onset.diff().dropna().to_numpy()
print('gaps: mean %.1f sd %.1f min %.1f max %.1f' % (gaps.mean(), gaps.std(), gaps.min(), gaps.max()))
print('KS vs U(150,300): p=%.3f; vs exponential(mean): p=%.4f; vs shifted exp (150+Exp): p=%.3f' % (
    stats.kstest(gaps, 'uniform', (150, 150)).pvalue, stats.kstest(gaps, 'expon', (0, gaps.mean())).pvalue,
    stats.kstest(gaps, 'expon', (150, gaps.mean() - 150)).pvalue))
print('gaps end->next onset', np.round(np.sort(gaps - 3), 1))
ins = r[r.op == 0]; k = ins[ins.ioc]
jt = ev.end.to_numpy()  # drought end ~ jump time
edges = np.arange(-1, 6.01, 0.25); prof = np.zeros(len(edges) - 1); prof_b = prof.copy()
for j in jt:
    d = k.t.to_numpy() - j
    prof += np.histogram(d, edges)[0]; prof_b += np.histogram(d[k.side.to_numpy() == 0], edges)[0]
prof /= len(jt) * 0.25; prof_b /= len(jt) * 0.25
print('taker rate (/s, both sides) aligned to drought end:')
for e, p, pb in zip(edges[:-1], prof, prof_b): print(f'  {e:+5.2f} {p:6.1f} buy {pb:5.1f}', '#' * int(p))
jump = np.array([at(mid, j + 0.5) - at(mid, j - 0.1) for j in jt])
nb = np.array([((k.t > j) & (k.t < j + 3.5)).sum() for j in jt])
print('jump sizes (mid[j+0.5]-mid[j-0.1]):', np.round(jump, 1).tolist())
print('|jump| mean %.1f sd(jump) %.1f; kurtosis %.2f' % (np.abs(jump).mean(), jump.std(), stats.kurtosis(jump)))
print('burst takers in (0,3.5)s vs |jump|: corr %.2f; slope %s' % (np.corrcoef(np.abs(jump), nb)[0, 1], np.round(np.polyfit(np.abs(jump), nb, 1), 2)))
for j, n, jj in sorted(zip(np.abs(jump), nb, jump)): pass
q = pd.DataFrame({'absjump': np.abs(jump), 'nburst': nb}).sort_values('absjump'); print(q.T.round(0).to_string())
# depth at touch in drought
bq = at(r.bid_qty_before.to_numpy(), 0)  # dummy
tq = (r.bid_qty_before + r.ask_qty_before).to_numpy() / 2
for lo, hi_ in [(-3, -0.5), (0.5, 1.5), (1.5, 2.9), (3.5, 6), (10, 20)]:
    vals = [np.median(tq[(tt > o + lo) & (tt < o + hi_)]) for o in ev.onset]
    print(f'median touch qty in [{lo},{hi_}) after onset: {np.mean(vals):.0f}')
# coincidence with volatile regime (posterior was fit with shock windows missing)
for lo, hi_ in [(-60, -20), (-20, -2), (20, 40), (40, 80)]:
    v = np.mean([hi[int(o + lo):int(o + hi_)].mean() for o in ev.onset])
    print(f'volatile occupancy in [{lo},{hi_}) around shocks: {v:.3f} (overall {hi.mean():.3f})')
