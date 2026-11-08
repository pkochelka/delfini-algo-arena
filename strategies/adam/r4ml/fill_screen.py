"""Decision ML screen: can a model tell, at fill time, which r4_best fills lose money?

    .venv/Scripts/python strategies/adam/r4ml/fill_screen.py [--npz logs/r4ml/state_r4.npz]

For each group (calm passive, calm IOC, turbulent passive, turbulent IOC) the target is the per-lot markout
side * (mid(t + h) - fill price), h = 1 s and 5 s, clipped to +-20 ticks (shock jumps). Features use only what the
strategy knows at the fill (filter state, book, recent flow/moves from the 20 ms state log, all signed by fill side).
Out-of-sample: 4 contiguous blocks, leave one out. Reports, per model, the lot-weighted markout of the fills the model
puts in its worst 10 % / 20 %, and the naive "skip value" = -sum(qty * markout) over fills predicted < 0 (what a
gate would save if skipping a fill had no other effect, which is an upper bound: the fill also cashes out position).
The hand rule to beat: edge vs F <= 0.25.
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
a = ap.parse_args()
z = np.load(ROOT / a.npz)
S = {k: z['x'][:, i] for i, k in enumerate(z['cols'])}
Fl = {k: z['fills'][:, i] for i, k in enumerate(z['fcols'])}
ts = S['t']
mid_s = 0.5 * (S['bb'] + S['ba'])


def at(arr, times, before=True):
    j = np.searchsorted(ts, times, side='left' if before else 'right') - (1 if before else 1)
    return arr[np.clip(j, 0, len(ts) - 1)]


t = Fl['t']
sd = Fl['side']
px = Fl['price']
qty = Fl['qty']
mid0 = 0.5 * (Fl['bb'] + Fl['ba'])
pos0 = Fl['pos0']  # position after this fill was applied? logged before orig handler -> before the fill
feat = {
    'edgeF': sd * (Fl['F'] - px),
    'edgeM': sd * (mid0 - px),
    'gap': sd * (Fl['F'] - mid0),
    'kv': sd * Fl['kv'],
    'pv': Fl['pv'], 'sdF': Fl['sd'], 'Y': sd * Fl['Y'],
    'opening': (sd * pos0 >= 0).astype(float),
    'abspos': np.abs(pos0) / 100,
    'spos': sd * pos0 / 100,
    'qty': np.log1p(qty),
    'age': np.log1p(np.nan_to_num(Fl['age'], nan=0.0)),
    'spread': Fl['ba'] - Fl['bb'],
    'imb_own': np.where(sd > 0, Fl['bq1'] - Fl['aq1'], Fl['aq1'] - Fl['bq1']) / np.maximum(Fl['bq1'] + Fl['aq1'], 1),
}
for w in (0.2, 0.5, 1, 2, 5):
    feat[f'dm{w}'] = sd * (at(mid_s, t) - at(mid_s, t - w))
    feat[f'dF{w}'] = sd * (at(S['F'], t) - at(S['F'], t - w))
for w in (0.2, 1, 5):
    feat[f'flow{w}'] = sd * (at(S['cum_sv'], t) - at(S['cum_sv'], t - w)) / 100
    feat[f'vol{w}'] = np.log1p(at(S['cum_v'], t) - at(S['cum_v'], t - w))
names = list(feat)
X = np.column_stack([feat[k] for k in names])
nb = 4
blk = np.minimum(((t - t.min()) / (t.max() - t.min() + 1e-9) * nb).astype(int), nb - 1)
groups = {
    'calm passive': (Fl['ioc'] == 0) & (Fl['turb'] == 0),
    'calm IOC': (Fl['ioc'] == 1) & (Fl['turb'] == 0),
    'turb passive': (Fl['ioc'] == 0) & (Fl['turb'] == 1),
    'turb IOC': (Fl['ioc'] == 1) & (Fl['turb'] == 1),
}


def summarize(name, score, y, q, m):
    """score: higher = better fill. Worst-decile markout and naive skip value of score < 0 (if a forecast)."""
    s, yy, qq = score[m], y[m], q[m]
    o = np.argsort(s)
    cq = np.cumsum(qq[o]) / qq.sum()
    w10 = o[cq <= 0.1]
    w20 = o[cq <= 0.2]
    mk = lambda ix: np.sum(yy[ix] * qq[ix]) / max(qq[ix].sum(), 1)  # noqa: E731
    return f'{name:>8}: worst10% {mk(w10):+.2f} worst20% {mk(w20):+.2f}'


for gname, gm in groups.items():
    for h in (1.0, 5.0):
        y = np.clip(sd * (at(mid_s, t + h, before=False) - px), -20, 20)
        ok = gm & (t + h < ts[-1])
        base = np.sum(y[ok] * qty[ok]) / qty[ok].sum()
        pred = {'ridge': np.full(len(y), np.nan), 'gbm': np.full(len(y), np.nan)}
        for b in range(nb):
            te = ok & (blk == b)
            tr = ok & (blk != b)
            m = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(X[tr], y[tr], ridge__sample_weight=qty[tr])
            pred['ridge'][te] = m.predict(X[te])
            m = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.03, max_depth=3, max_leaf_nodes=8,
                                              min_samples_leaf=300, l2_regularization=10.0)
            m.fit(X[tr], y[tr], sample_weight=qty[tr])
            pred['gbm'][te] = m.predict(X[te])
        out = [f'{gname:13} h={h:<3} n={ok.sum():5d} lots={qty[ok].sum() / 1e3:6.1f}k mean {base:+.2f}/lot |']
        out.append(summarize('edgeF', feat['edgeF'], y, qty, ok))
        for k, p_ in pred.items():
            r2 = 1 - np.sum(qty[ok] * (y[ok] - p_[ok]) ** 2) / np.sum(qty[ok] * (y[ok] - base) ** 2)
            neg = ok & (p_ < 0)
            skip = -np.sum(y[neg] * qty[neg])
            sk_b = [-np.sum((y * qty)[neg & (blk == b)]) for b in range(nb)]
            out.append(summarize(k, p_, y, qty, ok) + f' R2 {r2:+.3f} skip<0: {qty[neg].sum() / 1e3:.0f}k lots '
                       f'saves {skip / 1e3:+.1f}k (blocks ' + ','.join(f'{v / 1e3:+.0f}' for v in sk_b) + ')')
        print(' | '.join(out), flush=True)
