"""Lot-weighted markouts of our ladder fills by quote-time features (output of build_fills.py).

    uv run python research/round1/signals/fill_buckets.py FILLS.pkl
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

H = ['0.1', '1', '5', '30', '120']


def prep(f: pd.DataFrame) -> pd.DataFrame:
    f = f.copy()
    o = f.our.to_numpy()
    for h in H:
        f['mo' + h] = o * (f['mid_' + h] - f.price)
    f['mo_pre'] = o * (f.mid_pre - f.price)
    buy = o > 0
    # our side / other side, at t - 5 ms
    f['q_our'] = np.where(buy, f.qb_5, f.qa_5)
    f['q_opp'] = np.where(buy, f.qa_5, f.qb_5)
    f['share'] = f.q_our / (f.q_our + f.q_opp)
    f['q2_our'] = np.where(buy, f.qb2_5, f.qa2_5)
    f['d5_share'] = np.where(buy, f.db5_5 / (f.db5_5 + f.da5_5), f.da5_5 / (f.db5_5 + f.da5_5))
    f['spread'] = f.ba_5 - f.bb_5
    # where our fill price sat relative to the touch at t-5ms (0 = at touch; <0 behind)
    f['px_vs_touch'] = np.where(buy, f.price - f.bb_5, f.ba_5 - f.price)
    for lag in (20, 50, 200):
        same = np.where(buy, f[f'bb_{lag}'] == f.bb_5, f[f'ba_{lag}'] == f.ba_5)
        prev = np.where(buy, f[f'qb_{lag}'], f[f'qa_{lag}'])
        f[f'deplete_{lag}'] = np.where(same, f.q_our / prev.clip(1), np.nan)
    f['flow_vs_us_100'] = -o * f.flow_100  # >0: aggressors were trading against our side
    f['flow_vs_us_1000'] = -o * f.flow_1000
    micro = (f.ba_5 * f.qb_5 + f.bb_5 * f.qa_5) / (f.qb_5 + f.qa_5)
    f['micro_vs_px'] = o * (micro - f.price)  # >0 microprice on our favourable side
    return f


def table(f: pd.DataFrame, col: str, bins, label=None) -> pd.DataFrame:
    g = f.groupby(pd.cut(f[col], bins), observed=True)
    rows = []
    for k, d in g:
        w = d.quantity.to_numpy(float)
        r = {'bucket': str(k), 'fills': len(d), 'lots': int(w.sum()), 'lot%': 100 * w.sum() / f.quantity.sum()}
        for h in H:
            r['mo' + h] = np.average(d['mo' + h].fillna(0), weights=w)
        r['pnl30'] = (d.mo30.fillna(0) * w).sum()
        rows.append(r)
    t = pd.DataFrame(rows)
    print(f'\n## {label or col}')
    print(t.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
    return t


def main() -> None:
    f = prep(pd.read_pickle(sys.argv[1]))
    w = f.quantity.to_numpy(float)
    print('all lots', int(w.sum()), {h: round(np.average(f['mo' + h].fillna(0), weights=w), 3) for h in H})
    table(f, 'share', [0, .05, .1, .2, .3, .5, .7, .9, 1.0], 'our-side share of touch qty at t-5ms')
    table(f, 'q_our', [-1, 50, 100, 200, 400, 800, 1500, 3000, 1e9], 'market qty at our touch at t-5ms')
    table(f, 'deplete_50', [0, .25, .5, .75, .95, 1.05, 2, 1e9], 'our-touch qty now / 50ms earlier (same price)')
    table(f, 'deplete_200', [0, .25, .5, .75, .95, 1.05, 2, 1e9], 'our-touch qty now / 200ms earlier')
    table(f, 'spread', [0, 1, 2, 3, 100], 'spread at t-5ms')
    table(f, 'px_vs_touch', [-100, -2, -1, 0, 1, 100], 'fill price vs our touch at t-5ms (neg = behind)')
    table(f, 'flow_vs_us_100', [-1e9, -300, -50, -1, 0, 50, 300, 1e9], 'taker flow against our side, last 100ms')
    table(f, 'flow_vs_us_1000', [-1e9, -1000, -200, -1, 0, 200, 1000, 1e9], 'taker flow against our side, last 1s')
    table(f, 'nmove_1s', [-1, 0, 1, 3, 6, 1e9], 'mid changes in last 1 s')
    table(f, 'd5_share', [0, .2, .35, .5, .65, .8, 1.0], 'our-side share of 5-level depth')
    table(f, 'micro_vs_px', [-10, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 10], 'microprice minus our price (our dir)')
    table(f, 'taker_qty', [0, 100, 300, 800, 1600, 1e9], 'taker order size')
    # 2D: share x taker size
    f['share_b'] = pd.cut(f.share, [0, .1, .3, .7, 1])
    f['tq_b'] = pd.cut(f.taker_qty, [0, 300, 800, 1e9])
    piv = f.assign(x=f.mo30 * f.quantity).groupby(['share_b', 'tq_b'], observed=True).agg(lots=('quantity', 'sum'), x=('x', 'sum'))
    piv['mo30'] = piv.x / piv.lots
    print('\n## share x taker size (30 s)\n', piv[['lots', 'mo30']].to_string())


if __name__ == '__main__':
    main()
