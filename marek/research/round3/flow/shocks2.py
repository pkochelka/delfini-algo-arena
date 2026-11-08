"""Shock events: gap law, drought profile (maker/ladder/taker rates around onset), jump size, burst profile,
burst count vs |jump|, coincidence with volatile regime. Compare with round 2 shock_events.csv."""
import numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
D = Path(__file__).parent; C = D / 'cache'
r = pd.read_pickle(C / 'rows.pkl'); ev = pd.read_csv(D / 'shock_events.csv')
r2 = pd.read_csv(D.parents[1] / 'round2/flow/shock_events.csv')
reg = np.load(C / 'hmm_joint_2.npy')[:, 1] > 0.5
L = pd.read_pickle(C / 'ladders.pkl')
tt = r.t.to_numpy(); mid = ((r.bid_after + r.ask_after) / 2).ffill().to_numpy()
def at(a, x): return a[np.clip(np.searchsorted(tt, x, 'right') - 1, 0, None)]
for nm, e in [('r3', ev), ('r2', r2)]:
    g = e.onset.diff().dropna().to_numpy()
    print(f'{nm} n {len(e)} first {e.onset.iloc[0]:.1f} gaps mean {g.mean():.1f} sd {g.std():.1f} min {g.min():.1f} max {g.max():.1f} '
          f'KS U(150,300) p={stats.kstest(g, "uniform", (150, 150)).pvalue:.3f} KS exp p={stats.kstest(g, "expon", (0, g.mean())).pvalue:.4f} '
          f'KS 150+Exp p={stats.kstest(g, "expon", (150, g.mean() - 150)).pvalue:.3f}; dur {e.dur.mean():.3f}±{e.dur.std():.3f} rate {e.maker_rate.mean():.1f}')
g3 = ev.onset.diff().dropna().to_numpy(); g2 = r2.onset.diff().dropna().to_numpy()
print('pooled gaps r2+r3: KS U(150,300) p=%.3f, min %.1f; 2-sample KS r2 vs r3 p=%.3f' % (stats.kstest(np.r_[g2, g3], 'uniform', (150, 150)).pvalue, min(g2.min(), g3.min()), stats.ks_2samp(g2, g3).pvalue))
print('r3 gaps sorted', np.round(np.sort(g3), 1))
ins = r[r.op == 0]; k = ins[ins.ioc]; m = ins[~ins.ioc]
on = ev.onset.to_numpy(); jt = ev.end.to_numpy()
edges = np.arange(-2, 8.01, 0.25)
def prof(times, ref):
    h = np.zeros(len(edges) - 1)
    for j in ref: h += np.histogram(times - j, edges)[0]
    return h / len(ref) / 0.25
pt = prof(k.t.to_numpy(), on); pm = prof(m.t.to_numpy(), on); pl = prof(L.t0.to_numpy(), on)
pb = prof(k.t[k.side == 0].to_numpy(), on)
print('aligned on ONSET: maker /s, ladder starts /s, takers /s (buy)')
for e_, a, b, c, d in zip(edges[:-1], pm, pl, pt, pb): print(f'  {e_:+5.2f} mk {a:6.1f} lad {b:5.2f} tk {c:5.1f} buy {d:5.1f}', '#' * int(c))
jump = np.array([at(mid, j + 0.5) - at(mid, j - 0.1) for j in jt])
nb = np.array([((k.t > j) & (k.t < j + 3.5)).sum() for j in jt])
ev['jump_mid'] = jump; ev['burst_takers_3p5s'] = nb
ev['vol_before'] = [reg[int(o) - 10:int(o)].mean() for o in on]; ev['vol_after'] = [reg[int(o) + 6:int(o) + 16].mean() for o in on]
print('jumps', np.round(jump, 1).tolist())
print('|jump| mean %.1f sd %.1f kurt %.2f; r2 sd 43.5' % (np.abs(jump).mean(), jump.std(), stats.kurtosis(jump)))
print('burst takers vs |jump| corr %.2f slope %s' % (np.corrcoef(np.abs(jump), nb)[0, 1], np.round(np.polyfit(np.abs(jump), nb, 1), 2)))
for lo, hi_ in [(-60, -20), (-20, -2), (6, 20), (20, 60)]:
    v = np.mean([reg[int(o + lo):int(o + hi_)].mean() for o in on]); print(f'volatile occupancy [{lo},{hi_}) around onsets {v:.3f} (overall {reg.mean():.3f})')
ev.to_csv(D / 'shock_events.csv', index=False)
