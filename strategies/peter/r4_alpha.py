"""Strategy-side momentum study on a r4_record.py recording (100 ms grid).

    uv run python strategies/peter/r4_alpha.py logs/r4/rec_devbest.npz
1. how well past F moves / F - mid / taker flow predict the future F move at 1-30 s (OLS, first half fit,
   second half R2), calm vs turbulent
2. where the recorded strategy's PnL comes from: inventory carry (pos * d mid) vs the rest, and the carry split by
   whether the position agreed with the momentum forecast
3. a crude directional ceiling: hold +-100 in the forecast direction when |forecast| > thr, paying the half spread
"""
from __future__ import annotations

import sys

import numpy as np


def lag(x, k):  # x[t] - x[t-k]
    out = np.full(x.size, np.nan)
    out[k:] = x[k:] - x[:-k]
    return out


def lead(x, k):  # x[t+k] - x[t]
    out = np.full(x.size, np.nan)
    out[:-k] = x[k:] - x[:-k]
    return out


def fit(y, X, tr):
    m = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    A = np.column_stack([X, np.ones(len(y))])
    a, b = m & tr, m & ~tr
    beta, *_ = np.linalg.lstsq(A[a], y[a], rcond=None)
    r = y[b] - A[b] @ beta
    r2 = 1 - (r @ r) / ((y[b] - y[b].mean()) @ (y[b] - y[b].mean()))
    return beta, r2, A @ beta


def main() -> None:
    z = np.load(sys.argv[1])
    a, cols = z['a'], list(z['cols'])
    c = {k: a[:, i] for i, k in enumerate(cols)}
    t, F = c['t'], c['F']
    mid = 0.5 * (c['bb'] + c['ba'])
    turb = c['turb'] > 0.5
    tr = t < t[0] + 0.5 * (t[-1] - t[0])
    feats = {'dF.5': lag(F, 5), 'dF1': lag(F, 10), 'dF2': lag(F, 20), 'dF5': lag(F, 50), 'dF10': lag(F, 100),
             'gap': F - mid, 'q1': lag(c['cq'], 10) / 100, 'q5': lag(c['cq'], 50) / 100,
             'n1': lag(c['cn'], 10), 'n5': lag(c['cn'], 50)}
    names = list(feats)
    X = np.column_stack([feats[k] for k in names])
    print('features', names)
    for h in (10, 20, 50, 100, 300):
        y = lead(F, h)
        for lab, m in (('all', np.ones_like(turb)), ('calm', ~turb), ('turb', turb)):
            yy = np.where(m, y, np.nan)
            beta, r2, _ = fit(yy, X, tr)
            print(f'h {h / 10:4.1f}s {lab:4} R2oos {r2:+.3f}  ' + ' '.join(f'{n} {b:+.2f}' for n, b in zip(names, beta)))
    # single-feature slopes, all regimes, for intuition
    for k in ('dF1', 'dF5', 'gap', 'q5', 'n5'):
        print(k, ' '.join(f'{h / 10:.0f}s {np.polyfit(*_ok(feats[k], lead(F, h)), 1)[0]:+.2f}' for h in (10, 50, 100, 300)))
    # ---- PnL attribution of the recorded strategy
    pnl, pos = c['pnl'], c['pos']
    dp = np.diff(pnl)
    carry = pos[:-1] * np.diff(mid)
    print(f'\nPnL {pnl[-1] / 1e3:.1f}k: carry (pos*dmid) {carry.sum() / 1e3:.1f}k, rest (edge at fill) {(dp - carry).sum() / 1e3:.1f}k')
    for lab, m in (('calm', ~turb[:-1]), ('turb', turb[:-1])):
        print(f'  {lab}: carry {carry[m].sum() / 1e3:.1f}k rest {(dp - carry)[m].sum() / 1e3:.1f}k  |pos| mean {np.abs(pos[:-1][m]).mean():.0f}')
    _, _, pred = fit(lead(F, 50), X, tr)
    agree = np.sign(pos[:-1]) == np.sign(pred[:-1])
    print(f'  carry with pos agreeing with 5 s forecast {carry[agree].sum() / 1e3:.1f}k, against {carry[~agree & (pos[:-1] != 0)].sum() / 1e3:.1f}k')
    # ---- directional ceiling
    for h in (20, 50, 100):
        _, _, pred = fit(lead(F, h), X, tr)
        for thr in (1, 2, 4, 8):
            want = np.where(np.abs(pred) > thr, 100 * np.sign(pred), 0.0)
            want = np.nan_to_num(want)
            g = want[:-1] * np.diff(mid)
            cost = np.abs(np.diff(want)) * 0.5 * (c['ba'] - c['bb'])[1:]
            m2 = ~tr[:-1]
            print(f'dir h {h / 10:.0f}s thr {thr}: gross {g[m2].sum() / 1e3:.0f}k cost {cost[m2].sum() / 1e3:.0f}k '
                  f'(second half) time in {np.mean(want[~tr] != 0):.2f}')


def _ok(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]


if __name__ == '__main__':
    main()
