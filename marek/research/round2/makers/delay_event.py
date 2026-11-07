"""Cancel response delay, from an event study on taker impact jumps (exact timestamps).

For each IOC with a large impact (c*sqrt(filled) > thr, c = 0.03 calm / 0.13 vol), take the maker orders resting at the
taker's time on the side that the jump moves against (asks after a buy taker, bids after a sell), that survive the
taker (not filled by it), and measure their cancel-row intensity in 1-2 ms bins after the taker, vs the same orders'
intensity before it. The intensity rise after the jump traces the delay distribution (decision -> cancel row).
"""
import numpy as np, pandas as pd
from prep import load
import states

o = load()
g = o[~o.ioc]
f = o[o.ioc & (o.filled > 0)].copy()
st = states.state(f.t_us.to_numpy())
c = np.where(st == 'vol', 0.13, 0.03)
f['imp'] = c * np.sqrt(f.filled)
f['st'] = st
gt = g.t_us.to_numpy(); ge = g.cancel_us.to_numpy(); gs = g.sg.to_numpy()
order = np.argsort(gt)

bins = np.r_[-50, -20, -10, 0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 70, 100, 150, 200, 300] * 1000
for name, sel in [('calm imp>0.5', (f.st == 'calm') & (f.imp > 0.5)), ('vol imp>1.5', (f.st == 'vol') & (f.imp > 1.5)),
                  ('vol imp 0.5-1.5', (f.st == 'vol') & (f.imp > 0.5) & (f.imp <= 1.5))]:
    ev = f[sel]
    cnt = np.zeros(len(bins) - 1); expo = np.zeros(len(bins) - 1)
    for t0, sgn in zip(ev.t_us.to_numpy(), ev.sg.to_numpy()):
        # resting before t0, cancel row after t0 - 50 ms, adverse side = opposite side of the taker
        lo = np.searchsorted(gt, t0 - 5_000_000)
        hi = np.searchsorted(gt, t0)
        idx = np.arange(lo, hi)
        idx = idx[(gs[idx] == -sgn) & (ge[idx] > t0)]
        # exclude orders filled by this taker: they are at the touch; keep all - fill does not stop the cancel row
        rel_end = ge[idx] - t0
        rel_start = gt[idx] - t0
        for k in range(len(bins) - 1):
            a, b = bins[k], bins[k + 1]
            s_ = np.maximum(rel_start, a)
            e_ = np.minimum(rel_end, b)
            expo[k] += np.clip(e_ - s_, 0, None).sum() / 1e6
            cnt[k] += ((rel_end > a) & (rel_end <= b) & (rel_start < a)).sum()
    hz = cnt / np.maximum(expo, 1e-9)
    print(f'{name}: n events {len(ev)}, mean impact {ev.imp.mean():.2f}')
    print(' bin_ms  hazard/s  n')
    for k in range(len(bins) - 1):
        print(f'  {bins[k] / 1000:>5.0f}..{bins[k + 1] / 1000:<5.0f} {hz[k]:8.2f} {int(cnt[k]):6d}')
