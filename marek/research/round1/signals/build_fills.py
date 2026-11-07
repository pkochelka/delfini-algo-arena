"""Rebuild the simulated (market-only) book from a full-log run and attach quote-time features to our fills.

    uv run python research/round1/signals/build_fills.py logs/r1_ladder_mm OUT.pkl

book_deltas.csv -> per distinct timestamp: best bid/ask of *market* orders, market qty at the touch and
one level behind, our resting qty per side. trades.csv -> our fills (maker_owner == 'user').
Features are taken from the last book state at or before t_fill - LAT (5 ms: the latest state a
modify/cancel could still act on); markouts from the market mid after the fill.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

LAT = 5_000


def book_states(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path / 'book_deltas.csv', dtype={'side': 'string', 'owner': 'string'})
    ts = d.timestamp_us.to_numpy().tolist()
    side = (d.side == 'A').to_numpy().tolist()
    px = d.price.to_numpy().tolist()
    user = (d.owner != 'market').to_numpy().tolist()
    dq = d.qty_diff.to_numpy().tolist()
    mk = ({}, {})  # bid, ask: price -> market qty
    us = ({}, {})
    out = []
    n = len(ts)
    for i in range(n):
        s = side[i]
        book = us[s] if user[i] else mk[s]
        v = book.get(px[i], 0) + dq[i]
        if v:
            book[px[i]] = v
        else:
            book.pop(px[i], None)
        if i + 1 == n or ts[i + 1] != ts[i]:
            b, a = mk
            if b and a:
                bb, ba = max(b), min(a)
                ub = sum(us[0].values()); ua = sum(us[1].values())
                out.append((ts[i], bb, ba, b[bb], a[ba], b.get(bb - 1, 0), a.get(ba + 1, 0),
                            sum(q for p, q in b.items() if p >= bb - 4), sum(q for p, q in a.items() if p <= ba + 4),
                            ub, ua, us[0].get(bb, 0), us[1].get(ba, 0)))
    return pd.DataFrame(out, columns=['t', 'bb', 'ba', 'qb', 'qa', 'qb2', 'qa2', 'db5', 'da5',
                                      'ub', 'ua', 'ub_touch', 'ua_touch'])


def main() -> None:
    run, out = Path(sys.argv[1]), Path(sys.argv[2])
    q = book_states(run)
    q['mid'] = (q.bb + q.ba) / 2
    tr = pd.read_csv(run / 'trades.csv', dtype={'taker_order_id': 'string', 'maker_order_id': 'string',
                                               'taker_owner': 'string', 'maker_owner': 'string', 'taker_side': 'string'})
    # market taker flow (signed, + = buy aggressor), per timestamp
    mt = tr[tr.taker_owner == 'market']
    sgn = np.where(mt.taker_side == 'B', 1, -1)
    flow = pd.Series(sgn * mt.quantity.to_numpy(), index=mt.timestamp_us.to_numpy()).groupby(level=0).sum()
    ft, fc = flow.index.to_numpy(), np.concatenate([[0], np.cumsum(flow.to_numpy())])
    fa = np.concatenate([[0], np.cumsum(np.abs(flow.to_numpy()))])

    f = tr[tr.maker_owner == 'user'].copy()
    f['our'] = np.where(f.taker_side == 'B', -1, 1)  # +1 we bought
    # taker size (all fills of that taker order)
    tsz = tr.groupby('taker_order_id').quantity.sum()
    f['taker_qty'] = tsz.reindex(f.taker_order_id).to_numpy()
    # lots ahead of us: market fills at our price by the same taker, before our row
    tr['_row'] = np.arange(len(tr))
    f['_row'] = tr.loc[f.index, '_row']
    T = q.t.to_numpy()

    def at(times, col, strict_before=False):
        i = np.searchsorted(T, times, side='left' if strict_before else 'right') - 1
        v = q[col].to_numpy(dtype=float)[np.clip(i, 0, None)]
        return np.where(i >= 0, v, np.nan)

    t = f.timestamp_us.to_numpy()
    s = t - LAT
    for col in ['bb', 'ba', 'qb', 'qa', 'qb2', 'qa2', 'db5', 'da5', 'mid', 'ub', 'ua', 'ub_touch', 'ua_touch']:
        f[col + '_5'] = at(s, col)
    for lag in (20_000, 50_000, 200_000):
        for col in ['qb', 'qa', 'bb', 'ba']:
            f[f'{col}_{lag // 1000}'] = at(s - lag, col)
    f['mid_pre'] = at(t, 'mid', strict_before=True)
    for h in (0.1, 1, 5, 30, 120):
        f[f'mid_{h}'] = at(t + int(h * 1e6), 'mid')
    for w in (100_000, 1_000_000):
        i1 = np.searchsorted(ft, s, side='right'); i0 = np.searchsorted(ft, s - w, side='right')
        f[f'flow_{w // 1000}'] = fc[i1] - fc[i0]
        f[f'absflow_{w // 1000}'] = fa[i1] - fa[i0]
    # realized short vol: number of mid changes in last 1 s before s
    mids = q.mid.to_numpy(); chg = np.concatenate([[0], np.cumsum(np.diff(mids) != 0)])
    i1 = np.searchsorted(T, s, side='right') - 1; i0 = np.searchsorted(T, s - 1_000_000, side='right') - 1
    f['nmove_1s'] = chg[np.clip(i1, 0, None)] - chg[np.clip(i0, 0, None)]
    f.drop(columns=['_row']).to_pickle(out)
    q.to_pickle(out.with_suffix('.book.pkl'))
    print(len(f), 'fills', f.quantity.sum(), 'lots')


if __name__ == '__main__':
    main()
