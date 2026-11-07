"""2-D splits and a small regression of fill markouts on quote-time features (incremental value of each signal).

    uv run python research/round1/signals/fill_2d.py FILLS.pkl
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from fill_buckets import prep


def piv(f, a, abins, b, bbins, h='mo30'):
    g = f.assign(x=f[h].fillna(0) * f.quantity, A=pd.cut(f[a], abins), B=pd.cut(f[b], bbins))
    t = g.groupby(['A', 'B'], observed=True).agg(lots=('quantity', 'sum'), x=('x', 'sum'))
    t[h] = t.x / t.lots
    print(f'\n## {a} x {b} ({h})')
    print(t[['lots', h]].unstack('B').round(3).to_string())


def main() -> None:
    f = prep(pd.read_pickle(sys.argv[1]))
    f['abs_q'] = f.q_our
    piv(f, 'share', [0, .1, .25, .5, 1], 'deplete_200', [0, .5, .95, 1e9])
    piv(f, 'share', [0, .1, .25, .5, 1], 'flow_vs_us_100', [-1e9, 0, 50, 1e9])
    piv(f, 'share', [0, .1, .25, .5, 1], 'abs_q', [-1, 50, 200, 800, 1e9])
    piv(f, 'share', [0, .1, .25, .5, 1], 'q2_our', [-1, 200, 800, 1e9])
    piv(f, 'share', [0, .1, .25, .5, 1], 'abs_q', [-1, 50, 200, 800, 1e9], 'mo1')
    # OLS (lot-weighted) of mo1 and mo30 on features
    X = pd.DataFrame({
        'share_c': f.share - .5,
        'thin50': (f.q_our <= 50).astype(float),
        'deplete200_lt.5': (f.deplete_200 < .5).fillna(False).astype(float),
        'flow100_vs': np.clip(f.flow_vs_us_100, -500, 500) / 100,
        'flow1000_vs': np.clip(f.flow_vs_us_1000, -2000, 2000) / 1000,
        'spread2': (f.spread >= 2).astype(float),
        'd5share_c': f.d5_share - .5,
        'q2_our_k': np.clip(f.q2_our, 0, 5000) / 1000,
        'nmove': np.clip(f.nmove_1s, 0, 10),
    })
    X['const'] = 1.0
    w = f.quantity.to_numpy(float)
    for h in ('mo1', 'mo5', 'mo30'):
        y = f[h].to_numpy()
        ok = np.isfinite(y) & np.isfinite(X.to_numpy()).all(1)
        sw = np.sqrt(w[ok])
        beta, *_ = np.linalg.lstsq(X.to_numpy()[ok] * sw[:, None], y[ok] * sw, rcond=None)
        res = y[ok] - X.to_numpy()[ok] @ beta
        r2 = 1 - np.average(res**2, weights=w[ok]) / np.average((y[ok] - np.average(y[ok], weights=w[ok]))**2, weights=w[ok])
        print(f'\nOLS {h} R2={r2:.4f}  ' + '  '.join(f'{k}={v:+.3f}' for k, v in zip(X.columns, beta)))


if __name__ == '__main__':
    main()
