"""Round-4 dynamics of the hidden price, from the causal filter path written by r4_fpath.py.

    uv run python strategies/peter/r4_dyn.py logs/r4/fpath_round4_public.npz [logs/r4/fpath_round3_public.npz]

Each change of the filter mean F is booked as
    K  taker part: the change at an event with a filled market taker (the filter's impact kick)
    X  exogenous part: every other change (maker inserts moving the posterior = S moves no taker explains,
       plus the transient decay)
Per bin (default 0.5 s) it prints autocorrelations and cross-correlations of K, X, the taker flow, and the
regression of future X on past X / K / flow / (F - mid): which of them carries round 4's momentum.
"""
from __future__ import annotations

import sys

import numpy as np


def bins(d: dict, dt: float):
    t = d['t'] * 1e-6
    F = d['F']
    dF = np.diff(F, prepend=F[0])
    q = d['q']
    is_k = q != 0
    b = ((t - t[0]) // dt).astype(np.int64)
    n = b[-1] + 1
    K = np.bincount(b, weights=np.where(is_k, dF, 0.0), minlength=n)
    X = np.bincount(b, weights=np.where(is_k, 0.0, dF), minlength=n)
    sgn = np.bincount(b, weights=np.sign(q), minlength=n)
    vol = np.bincount(b, weights=q, minlength=n)
    # end-of-bin state
    last = np.searchsorted(b, np.arange(n), side='right') - 1
    last = np.maximum(last, 0)
    Fe = F[last]
    mid = 0.5 * (d['bb'][last] + d['ba'][last])
    pv = d['pv'][last]
    return dict(K=K, X=X, sgn=sgn, vol=vol, F=Fe, mid=mid, pv=pv, gap=Fe - mid)


def acf(x, lags):
    x = x - x.mean()
    v = (x * x).mean()
    return [float((x[:-k] * x[k:]).mean() / v) for k in lags]


def xcorr(x, y, lags):
    """corr(x_t, y_{t+k})"""
    x = (x - x.mean()) / x.std()
    y = (y - y.mean()) / y.std()
    return [float((x[:-k] * y[k:]).mean()) for k in lags]


def fwd(x, h):
    c = np.cumsum(np.concatenate([[0.0], x]))
    out = np.full(x.size, np.nan)
    out[:-h] = c[1 + h:] - c[1:-h]  # sum of x_{t+1..t+h}
    return out


def back(x, h):
    c = np.cumsum(np.concatenate([[0.0], x]))
    out = np.full(x.size, np.nan)
    out[h - 1:] = c[h:] - c[:-h]  # sum of x_{t-h+1..t}
    return out


def ols(y, cols, names):
    m = np.isfinite(y)
    for c in cols:
        m &= np.isfinite(c)
    A = np.column_stack([c[m] for c in cols] + [np.ones(m.sum())])
    beta, *_ = np.linalg.lstsq(A, y[m], rcond=None)
    r = y[m] - A @ beta
    s2 = r @ r / (m.sum() - A.shape[1])
    se = np.sqrt(np.diag(s2 * np.linalg.inv(A.T @ A)))
    r2 = 1 - (r @ r) / ((y[m] - y[m].mean()) @ (y[m] - y[m].mean()))
    return '  '.join(f'{n} {b:+.3f} (t {b / s:+.1f})' for n, b, s in zip(names, beta, se)) + f'  R2 {r2:.3f}'


def report(path: str, dt: float = 0.5) -> None:
    d = dict(np.load(path))
    B = bins(d, dt)
    K, X = B['K'], B['X']
    print(f'==== {path}  bin {dt}s  n {K.size}')
    print(f'sd per bin: K {K.std():.3f}  X {X.std():.3f}  total {(K + X).std():.3f}')
    L = [1, 2, 4, 10, 20, 60]
    print('lags (bins)', L)
    print('ACF K+X  ', np.round(acf(K + X, L), 3))
    print('ACF X    ', np.round(acf(X, L), 3))
    print('ACF K    ', np.round(acf(K, L), 3))
    print('ACF sgn  ', np.round(acf(B['sgn'], L), 3))
    print('X -> fut X  ', np.round(xcorr(X, X, L), 3))
    print('K -> fut X  ', np.round(xcorr(K, X, L), 3))
    print('X -> fut sgn', np.round(xcorr(X, B['sgn'], L), 3))
    print('sgn -> fut X', np.round(xcorr(B['sgn'], X, L), 3))
    print('gap -> fut X', np.round(xcorr(B['gap'], X, L), 3))
    for calm in (True, False):
        m = (B['pv'] < 0.5) if calm else (B['pv'] >= 0.5)
        print(f'--- {"calm" if calm else "volatile"} share {m.mean():.2f}')
        for hb, hf in ((2, 2), (4, 4), (10, 10), (20, 20)):
            y = fwd(K + X, hf)
            pX, pK, ps = back(X, hb), back(K, hb), back(B['sgn'], hb)
            y = np.where(m, y, np.nan)
            print(f'  fut dF {hf * dt:4.1f}s ~ past {hb * dt:4.1f}s:',
                  ols(y, [pX, pK, ps, B['gap']], ['X', 'K', 'sgn', 'gap']))


if __name__ == '__main__':
    for p in sys.argv[1:]:
        report(p)
