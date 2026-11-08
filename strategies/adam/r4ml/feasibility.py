"""ML feasibility: can small models predict the next move of r4_best's fair value F beyond what its filter knows?

    .venv/Scripts/python strategies/adam/r4ml/feasibility.py [--npz logs/r4ml/state_r4.npz] [--step 0.1]

Target: F(t + h) - F(t) (F is the filter's posterior mean of S, a cleaner proxy of S than the mid) and
mid(t + h) - F(t) (what a take at the touch would mark against). Out-of-sample R^2 with 4 contiguous time blocks,
leave-one-block-out, a 10 s purge around the test block. Rows sampled every --step s.

Model ladder (each row adds to the one above):
  base   kv*h                    (what r4_best already predicts: the Kalman velocity drift)
  fstate ridge on the filter state (F - mid, kv, pv, sd, Y, turb, pos)
  all_lin ridge on everything (+ lagged F / mid moves, flow windows, imbalance, inserts, spread)
  gbm    HistGradientBoosting on everything
"""
import argparse
import pathlib

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = pathlib.Path(__file__).resolve().parents[3]
ap = argparse.ArgumentParser()
ap.add_argument('--npz', default='logs/r4ml/state_r4.npz')
ap.add_argument('--step', type=float, default=0.1)
ap.add_argument('--hs', default='0.2,0.5,1,2,5')
ap.add_argument('--gbm', type=int, default=1)
ap.add_argument('--clip', type=float, default=10.0, help='drop rows with |target| >= clip (shock jumps)')
a = ap.parse_args()

z = np.load(ROOT / a.npz)
X0, cols = z['x'], list(z['cols'])
c = {k: X0[:, i] for i, k in enumerate(cols)}
t = c['t']
# resample to a regular grid by last-observation (rows are logged at >= 20 ms spacing)
grid = np.arange(t[0] + 10.0, t[-1] - 10.0, a.step)
idx = np.searchsorted(t, grid, side='right') - 1
g = {k: v[idx] for k, v in c.items()}
g['t'] = grid
mid = 0.5 * (g['bb'] + g['ba'])
F = g['F']


def at(arr_full, times):
    j = np.searchsorted(t, times, side='right') - 1
    return arr_full[np.clip(j, 0, len(t) - 1)]


mid_full = 0.5 * (c['bb'] + c['ba'])
feat = {
    'gap': F - mid, 'kv': g['kv'], 'pv': g['pv'], 'sd': g['sd'], 'Y': g['Y'], 'turb': g['turb'], 'pos': g['pos'] / 100,
    'spread': g['ba'] - g['bb'],
    'imb1': (g['bq1'] - g['aq1']) / np.maximum(g['bq1'] + g['aq1'], 1),
    'imb3': (g['bq1'] + g['bq2'] + g['bq3'] - g['aq1'] - g['aq2'] - g['aq3'])
            / np.maximum(g['bq1'] + g['bq2'] + g['bq3'] + g['aq1'] + g['aq2'] + g['aq3'], 1),
    'ldepth': np.log1p(g['bq1'] + g['aq1']),
}
for w in (0.1, 0.5, 1, 2, 5, 15):
    feat[f'dF{w}'] = F - at(c['F'], grid - w)
    feat[f'dm{w}'] = mid - at(mid_full, grid - w)
for w in (0.2, 1, 5, 15):
    feat[f'flow{w}'] = (g['cum_sv'] - at(c['cum_sv'], grid - w)) / 100
    feat[f'vol{w}'] = np.log1p(g['cum_v'] - at(c['cum_v'], grid - w))
    feat[f'ins{w}'] = (g['cum_ins_b'] - at(c['cum_ins_b'], grid - w)) - (g['cum_ins_a'] - at(c['cum_ins_a'], grid - w))
    feat[f'big{w}'] = g['cum_big'] - at(c['cum_big'], grid - w)
names = list(feat)
XA = np.column_stack([feat[k] for k in names])
FS = ['gap', 'kv', 'pv', 'sd', 'Y', 'turb', 'pos']
XF = np.column_stack([feat[k] for k in FS])

nb = 4
blk = np.minimum((np.arange(len(grid)) * nb) // len(grid), nb - 1)
vol = g['pv'] > 0.5
print(f'rows {len(grid)}, features {len(names)}, volatile share {vol.mean():.2f}')


def r2(y, p, m):
    y, p = y[m], p[m]
    return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


for h in [float(x) for x in a.hs.split(',')]:
    Ff = at(c['F'], grid + h)
    mf = at(mid_full, grid + h)
    for tname, y in (('dF', Ff - F), ('mid-F', mf - F)):
        ok = np.isfinite(y) & (grid + h < t[-1]) & (np.abs(y) < a.clip)
        preds = {k: np.full(len(y), np.nan) for k in ('base', 'fstate', 'all_lin', 'gbm')}
        for b, rm in [(b, rm) for b in range(nb) for rm in (vol, ~vol)]:
            te0 = (blk == b) & ok
            lo, hi = grid[te0].min() - 10, grid[te0].max() + 10 + h
            te = te0 & rm
            tr = ok & rm & ~((grid >= lo) & (grid <= hi))
            # base: OLS slope on kv*h (+ intercept)
            xb = (g['kv'] * h)[:, None]
            m = Ridge(alpha=1e-6).fit(xb[tr], y[tr]); preds['base'][te] = m.predict(xb[te])
            m = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(XF[tr], y[tr]); preds['fstate'][te] = m.predict(XF[te])
            m = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(XA[tr], y[tr]); preds['all_lin'][te] = m.predict(XA[te])
            if a.gbm:
                m = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.03, max_depth=3, max_leaf_nodes=8,
                                                  min_samples_leaf=1000, l2_regularization=10.0)
                m.fit(XA[tr], y[tr]); preds['gbm'][te] = m.predict(XA[te])
        line = [f'h={h:<4} {tname:6}']
        for regime, mm in (('calm', ok & ~vol), ('vol', ok & vol)):
            vals = ' '.join(f'{k}={r2(y, preds[k], mm):+.3f}' for k in preds if np.isfinite(preds[k][mm]).all())
            per_blk = [r2(y, preds['gbm' if a.gbm else 'all_lin'], mm & (blk == b)) - r2(y, preds['fstate'], mm & (blk == b))
                       for b in range(nb)]
            line.append(f'| {regime}: {vals} | best-fstate per block ' + ','.join(f'{d:+.3f}' for d in per_blk))
        print(' '.join(line), flush=True)
