"""Summaries of an instrument.py pickle: queue position at activation, fills and markouts by queue
position and by placement, and touch transitions (how fast / how deep we get in at a new touch).

    uv run python research/round1/queue/analyze.py research/round1/queue/out/base.pkl
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from analysis import features as F, load_tape  # noqa: E402

pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 30)


def load(path: str):
    runs = pickle.load(open(path, 'rb'))
    orders, fills, trans = [], [], []
    for r in runs:
        o = pd.DataFrame.from_dict(r['orders'], orient='index')
        o['fold'] = r['fold']
        orders.append(o)
        f = pd.DataFrame(r['fills'], columns=['t', 'oid', 'side', 'price', 'qty', 'ahead'])
        f['fold'] = r['fold']
        fills.append(f)
        t = pd.DataFrame(r['trans'])
        t['fold'] = r['fold']
        trans.append(t)
    return pd.concat(orders), pd.concat(fills, ignore_index=True), pd.concat(trans, ignore_index=True), runs


def markout(tape, f: pd.DataFrame, hs=(1.0, 5.0, 30.0)) -> pd.DataFrame:
    sgn = np.where(f['side'] == 'B', 1, -1)
    for h in hs:
        mid = F.asof(tape, f['t'].to_numpy() / 1e6 + h, 'mid')
        f[f'mo{int(h)}'] = (mid - f['price']) * sgn
    return f


def wmean(g, c):
    return np.average(g[c], weights=g['qty']) if len(g) else np.nan


def main() -> None:
    path = sys.argv[1]
    orders, fills, trans, runs = load(path)
    print('PnL total', sum(r['pnl'] for r in runs))
    tape = load_tape(ROOT / 'data/round1.csv', verbose=False)
    fills = markout(tape, fills)
    lim = orders[orders['type'] == 'limit'].copy()
    lim['lat_ms'] = (lim['act_t'] - lim['sent_t']) / 1e3
    print('\n== own limit orders ==')
    print('n', len(lim), 'never active', lim['act_t'].isna().sum(), 'activation latency ms quantiles',
          lim['lat_ms'].quantile([.5, .9, .99]).round(2).tolist())
    print('rel to market touch at activation (+ inside, 0 at, - behind):')
    print(lim['rel_act'].value_counts().sort_index())
    lim['fill_frac'] = lim['filled'] / lim['qty']
    bins = [-1, 0, 200, 500, 1000, 2000, 5000, 1e9]
    lim['ahead_bin'] = pd.cut(lim['ahead_act'], bins)
    g = lim.groupby('ahead_bin', observed=True).agg(n=('qty', 'size'), qty=('qty', 'sum'), filled=('filled', 'sum'))
    g['fill_rate'] = g['filled'] / g['qty']
    print('\nby market lots ahead at activation:\n', g)
    print('\nend reasons:', lim['end'].value_counts(dropna=False).to_dict())
    life = (lim['end_t'] - lim['act_t']) / 1e6
    print('active life s quantiles', life.quantile([.1, .5, .9]).round(3).tolist())

    print('\n== fills ==')
    print('fills', len(fills), 'lots', fills['qty'].sum())
    fills['ahead_bin'] = pd.cut(fills['ahead'], bins)
    g = fills.groupby('ahead_bin', observed=True).apply(
        lambda g: pd.Series({'lots': g['qty'].sum(), 'mo1': wmean(g, 'mo1'), 'mo5': wmean(g, 'mo5'),
                             'mo30': wmean(g, 'mo30')}), include_groups=False)
    print('by market lots ahead at the last snapshot before the fill:\n', g.round(3))
    lk = lim[['rel_act', 'fold']].rename_axis('oid').reset_index()
    j = fills.merge(lk, on=['fold', 'oid'], how='left')
    g = j.groupby('rel_act').apply(lambda g: pd.Series({'lots': g['qty'].sum(), 'mo1': wmean(g, 'mo1'),
                                                         'mo30': wmean(g, 'mo30')}), include_groups=False)
    print('by placement rel to touch at activation:\n', g.round(3))

    print('\n== touch transitions ==')
    t = trans
    t['kind'] = np.where(t['dir'] > 0, 'improve', 'worsen')
    t['cause'] = np.where(t['traded_old'] > 0, 'traded', 'cancel/other')
    t['joined'] = t['join_t'].notna()
    t['delay_ms'] = (t['join_t'] - t['t']) / 1e3
    t['pre'] = t['have_q'] > 0
    # blocked: the side adds to a position already at the cap, so it cannot quote at all
    t['room0'] = np.where(t['side'] == 'B', t['pos'] >= 100, t['pos'] <= -100)
    print(t.groupby(['kind', 'cause']).size())
    print('\nby kind / room<=0 at the transition:')
    print(t.groupby(['kind', 'room0']).agg(n=('t', 'size'), joined=('joined', 'mean'), pre=('pre', 'mean'),
                                          delay_med=('delay_ms', 'median'), delay_p90=('delay_ms', lambda x: x.quantile(.9)),
                                          ahead_med=('join_ahead', 'median'), mq_new=('mq_new', 'median')).round(3))
    print('\n|abs dir| distribution', t['dir'].abs().value_counts().to_dict())
    # time on the new level before the next transition on that side
    t = t.sort_values(['fold', 'side', 't'])
    t['next_t'] = t.groupby(['fold', 'side'])['t'].shift(-1)
    t['dwell_s'] = (t['next_t'] - t['t']) / 1e6
    print('dwell of a touch level s, quantiles', t['dwell_s'].quantile([.1, .25, .5, .75, .9]).round(3).tolist())
    pos = pd.Series([p for r in runs for p in [x['pos'] for x in r['trans']]])
    print('|pos| at transitions: share ==100', (pos.abs() >= 100).mean().round(3), 'share 0', (pos == 0).mean().round(3))


if __name__ == '__main__':
    main()
