"""Latency / cancel-race view of the share signal and the fresh-vs-depleted level split.

For each of our fills: the onset of the current 'thin' run on our side (market share of the touch < THR,
spread 1) before the fill; fills whose run started < 5 ms before the fill cannot be dodged.
Also the level's depletion: market touch qty at t-5ms / its peak since that price became the touch.

    uv run python research/round1/signals/race.py FILLS.pkl     (FILLS.book.pkl next to it)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from fill_buckets import prep


def side_series(q: pd.DataFrame, buy: bool):
    px = (q.bb if buy else q.ba).to_numpy()
    qq = (q.qb if buy else q.qa).to_numpy().astype(float)
    share = qq / (q.qb + q.qa).to_numpy()
    run = np.concatenate([[0], np.cumsum(px[1:] != px[:-1])])
    peak = pd.Series(qq).groupby(run).cummax().to_numpy()
    start = pd.Series(q.t.to_numpy()).groupby(run).transform('first').to_numpy()
    return share, peak, qq, start


def main() -> None:
    fp = Path(sys.argv[1])
    f = prep(pd.read_pickle(fp))
    q = pd.read_pickle(fp.with_suffix('.book.pkl'))
    T = q.t.to_numpy()
    spread1 = (q.ba - q.bb).to_numpy() == 1
    out = {}
    for thr in (0.1, 0.2):
        onset = np.full(len(f), np.nan)
        dep = np.full(len(f), np.nan)
        age = np.full(len(f), np.nan)
        for buy in (True, False):
            share, peak, qq, start = side_series(q, buy)
            thin = (share < thr) & spread1
            # index of the last non-thin state at or before each index
            idx = np.where(~thin, np.arange(len(T)), -1)
            last_ok = np.maximum.accumulate(idx)
            m = (f.our > 0).to_numpy() == buy
            t = f.timestamp_us.to_numpy()[m]
            i_pre = np.searchsorted(T, t, side='left') - 1  # state just before the fill
            lo = last_ok[i_pre]
            on = np.where(thin[i_pre], T[np.clip(lo + 1, 0, len(T) - 1)], np.nan)
            onset[m] = t - on
            i5 = np.searchsorted(T, t - 5000, side='right') - 1
            dep[m] = qq[i5] / np.maximum(peak[i5], 1)
            age[m] = (t - start[i5]) / 1000
        f[f'onset_{thr}'] = onset / 1000  # ms the side had been thin before the fill (nan = not thin)
        f['dep'] = dep
        f['lvl_age_ms'] = age
    w = f.quantity
    for thr in (0.1, 0.2):
        c = f[f'onset_{thr}']
        b = pd.cut(c, [-1, 5, 20, 100, 1000, 1e9])
        t = f.assign(x=f.mo30 * w, x1=f.mo1 * w).groupby(b, observed=True).agg(lots=('quantity', 'sum'), x=('x', 'sum'), x1=('x1', 'sum'))
        t['mo1'] = t.x1 / t.lots
        t['mo30'] = t.x / t.lots
        print(f'\n## fills while our side thin (share < {thr}) just before the fill, by how long it had been thin (ms)')
        print(t[['lots', 'mo1', 'mo30', 'x']].round(3).to_string())
        print('not thin:', int(w[c.isna()].sum()), 'lots, mo30', round(np.average(f.mo30[c.isna()].fillna(0), weights=w[c.isna()]), 3))
    t = f.assign(x=f.mo30 * w, x1=f.mo1 * w, S=pd.cut(f.share, [0, .1, .2, .35, 1]), D=pd.cut(f.dep, [-.01, .3, .5, .8, 1.0]))
    g = t.groupby(['S', 'D'], observed=True).agg(lots=('quantity', 'sum'), x=('x', 'sum'), x1=('x1', 'sum'))
    g['mo1'] = g.x1 / g.lots
    g['mo30'] = g.x / g.lots
    print('\n## share x depletion (qty / peak since the price became the touch), t-5ms')
    print(g[['lots', 'mo1', 'mo30', 'x']].round(3).to_string())
    t = f.assign(x=f.mo30 * w, A=pd.cut(f.lvl_age_ms, [-1, 10, 50, 200, 1000, 5000, 1e9]))
    g = t.groupby('A', observed=True).agg(lots=('quantity', 'sum'), x=('x', 'sum'))
    g['mo30'] = g.x / g.lots
    print('\n## by age of our touch level at t-5ms (ms since that price became the touch)')
    print(g.round(3).to_string())


if __name__ == '__main__':
    main()
