"""Offline: how well do causal features nowcast S(t+h) - F(t) in volatile? (r4_vlog.py npz + oracle truth)
    uv run python strategies/peter/r4_vnow.py logs/r4/vlog_volsig.npz"""
import sys
import numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analysis.cache import load_tape
x = np.load(sys.argv[1])['x']
t, F, sd, pv, turb, pos, bb, ba, kv, kPv, Y, exc, sg = x.T
z = np.load(ROOT / 'logs/r4/oracle_truth.npz')
def S_at(tt):
    return z['S'][np.minimum(np.searchsorted(z['t'], tt, 'right'), len(z['S']) - 1)]
vol = np.zeros(len(t), bool)
for a, b in zip(z['va'], z['vb']): vol |= (t >= a) & (t < b)
sh = np.zeros(len(t), bool)
for o in z['on']: sh |= (t >= o - 1) & (t < o + 20)
mid = (bb + ba) / 2
o = load_tape(ROOT / 'data/round4.csv').orders
o = o[o.ioc & (o.filled > 0)]
tk_t = o.t_us.to_numpy() * 1e-6; tk_k = np.where(o.side.to_numpy() == 'B', 1.0, -1.0) * o.filled.to_numpy() ** 0.64
cum = np.r_[0, np.cumsum(tk_k)]
def flow(w):
    return cum[np.searchsorted(tk_t, t, 'right')] - cum[np.searchsorted(tk_t, t - w, 'right')]
def ema(v, tau):
    out = np.empty_like(v); a = 1 - np.exp(-0.01 / tau); m = v[0]
    for i, f in enumerate(v): m += a * (f - m); out[i] = m
    return out
def lagd(v, w):  # v(t) - v(t - w) on the 10 ms grid
    j = np.searchsorted(t, t - w); return v - v[np.maximum(j, 0)]
feats = {'kv': np.clip(kv, -10, 10), 'F-mid': np.clip(F - mid, -5, 5), 'dF.25': lagd(F, .25), 'dF.5': lagd(F, .5), 'dF1': lagd(F, 1.0),
         'dmid.5': lagd(mid, .5), 'flow.5': flow(.5) * 0.05, 'flow2': flow(2.0) * 0.05, 'Y': np.clip(Y, -30, 30),
         'kv*sd': np.clip(kv, -10, 10) * sd, 'kvconf': np.clip(kv, -10, 10) * np.sqrt(np.maximum(kPv, 1e-6))}
m = vol & ~sh & (sg == 0) & (turb > 0)
half = t < 3000
print('vol rows', m.sum(), 'rmse F-S', np.sqrt(np.mean((F - S_at(t))[m] ** 2)).round(3))
for h in (0.0, 0.15, 0.3, 0.5):
    y = (S_at(t + h) - F)[m]
    cur = y - 0.15 * feats['kv'][m]
    print(f'\nh {h}: var {y.var():.3f}  current lean 0.15 kv -> R2 {1 - cur.var() / y.var():+.3f}')
    for sets in (['kv'], ['kv', 'F-mid'], ['kv', 'dF.25', 'dF.5', 'dF1'], ['kv', 'F-mid', 'dF.25', 'dF.5', 'dF1', 'dmid.5'],
                 ['kv', 'F-mid', 'dF.25', 'dF.5', 'dF1', 'dmid.5', 'flow.5', 'flow2', 'Y'], ['kv', 'kvconf', 'kv*sd']):
        X = np.c_[[feats[k][m] for k in sets]].T; X = np.c_[X, np.ones(len(X))]
        hh = half[m]; r2 = []
        bs = []
        for tr, te in ((hh, ~hh), (~hh, hh)):
            b = np.linalg.lstsq(X[tr], y[tr], rcond=None)[0]; bs.append(b)
            r2.append(1 - (y[te] - X[te] @ b).var() / y[te].var())
        print(f'  {"+".join(sets):55s} oosR2 {r2[0]:+.3f} {r2[1]:+.3f}  b ' + ' '.join(f'{k}={v:+.3f}' for k, v in zip(sets, (bs[0] + bs[1]) / 2)))
