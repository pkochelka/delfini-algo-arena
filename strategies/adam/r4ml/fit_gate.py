"""Fit the turbulent-IOC gate: ridge on live-computable features -> per-lot 5 s markout. Two models, one per tape half,
so a backtest fold always uses the model fitted on the other half (strategy picks by its clock, split at T_SPLIT).

    .venv/Scripts/python strategies/adam/r4ml/fit_gate.py [--h 5] [--alpha 10]  -> prints the PARAMS dict for ml_gate.py
"""
import argparse
import json
import pathlib

import numpy as np
from sklearn.linear_model import Ridge

ROOT = pathlib.Path(__file__).resolve().parents[3]
ap = argparse.ArgumentParser()
ap.add_argument('--h', type=float, default=5.0)
ap.add_argument('--alpha', type=float, default=10.0)
ap.add_argument('--group', default='turb_ioc', choices=['turb_ioc', 'calm_pass'])
a = ap.parse_args()
T_SPLIT = 3000.0
GF = ['edgeF', 'gap', 'kv', 'Y', 'spos', 'opening', 'sdF', 'pv', 'dm0.5', 'dm1', 'dm2', 'dm5', 'dF0.5', 'dF1', 'dF2', 'dF5']
z = np.load(ROOT / 'logs/r4ml/state_r4.npz')
S = {k: z['x'][:, i] for i, k in enumerate(z['cols'])}
Fl = {k: z['fills'][:, i] for i, k in enumerate(z['fcols'])}
ts = S['t']
mid_s = 0.5 * (S['bb'] + S['ba'])


def at(arr, times, before=True):
    j = np.searchsorted(ts, times, side='left' if before else 'right') - 1
    return arr[np.clip(j, 0, len(ts) - 1)]


t, sd, px, qty, pos0 = Fl['t'], Fl['side'], Fl['price'], Fl['qty'], Fl['pos0']
mid0 = 0.5 * (Fl['bb'] + Fl['ba'])
feat = {'edgeF': sd * (Fl['F'] - px), 'gap': sd * (Fl['F'] - mid0), 'kv': sd * Fl['kv'], 'Y': sd * Fl['Y'],
        'spos': sd * pos0 / 100, 'opening': (sd * pos0 >= 0).astype(float), 'sdF': Fl['sd'], 'pv': Fl['pv']}
for w in (0.5, 1, 2, 5):
    feat[f'dm{w}'] = sd * (at(mid_s, t) - at(mid_s, t - w))
    feat[f'dF{w}'] = sd * (at(S['F'], t) - at(S['F'], t - w))
X = np.column_stack([feat[k] for k in GF])
y = np.clip(sd * (at(mid_s, t + a.h, before=False) - px), -20, 20)
if a.group == 'turb_ioc':
    g = (Fl['ioc'] == 1) & (Fl['turb'] == 1)
else:
    g = (Fl['ioc'] == 0) & (Fl['turb'] == 0)
g &= t + a.h < ts[-1]
models = {}
for name, tr in (('A', g & (t < T_SPLIT)), ('B', g & (t >= T_SPLIT))):
    mu, sg = X[tr].mean(0), X[tr].std(0) + 1e-9
    r = Ridge(alpha=a.alpha).fit((X[tr] - mu) / sg, y[tr], sample_weight=qty[tr])
    w = r.coef_ / sg
    b = r.intercept_ - np.sum(w * mu)
    models[name] = (w, b)
# out-of-sample check: model A on half 2, B on half 1
for name, te in (('A->h2', g & (t >= T_SPLIT)), ('B->h1', g & (t < T_SPLIT))):
    w, b = models[name[0]]
    p = X[te] @ w + b
    for thr in (-2, -1, 0, 1):
        m = p < thr
        print(f'{name} thr {thr:+}: skip {qty[te][m].sum() / 1e3:5.1f}k of {qty[te].sum() / 1e3:.1f}k lots, '
              f'mk {np.average(y[te][m], weights=qty[te][m]) if m.any() else 0:+.2f}, '
              f'naive save {-np.sum(y[te][m] * qty[te][m]) / 1e3:+.1f}k')
print(json.dumps({'mlg_feats': GF, 'mlg_wA': [round(v, 5) for v in models['A'][0]], 'mlg_bA': round(models['A'][1], 5),
                  'mlg_wB': [round(v, 5) for v in models['B'][0]], 'mlg_bB': round(models['B'][1], 5)}))
