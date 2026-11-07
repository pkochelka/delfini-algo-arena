"""Order table for round 2: one row per insert with the book just before it, cancel row time, taker fills.

Writes cache/orders.pkl. Columns: t_us, side(+1 buy/-1 sell), price, volume, ioc, filled, end, end_t_us,
cancel_us (cancel row time, NaN if none), bid, ask (touch just before insert), bid_qty, ask_qty.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from analysis.cache import load_tape  # noqa: E402

OUT = Path(__file__).parent / 'cache'


def build():
    tape = load_tape(ROOT / 'data/round2.csv')
    ev, o = tape.events, tape.orders.copy()
    prev = np.maximum(o.row.to_numpy() - 1, 0)
    first = o.row.to_numpy() == 0
    for k in ('bid', 'ask', 'bid_qty', 'ask_qty'):
        v = ev[k].to_numpy()[prev].astype(float)
        if k in ('bid', 'ask'):
            v[(v == 0)] = np.nan
        v[first] = np.nan
        o[k] = v
    c = ev.loc[ev.op == 'C', ['t_us', 'order_id']].rename(columns={'t_us': 'cancel_us'})
    o = o.merge(c, on='order_id', how='left')
    o['sg'] = np.where(o.side == 'B', 1, -1).astype(np.int8)
    o = o[['row', 't_us', 'sg', 'price', 'volume', 'ioc', 'filled', 'end', 'end_t_us', 'cancel_us',
           'bid', 'ask', 'bid_qty', 'ask_qty', 'queue_ahead', 'dist']]
    o['end'] = o['end'].astype(str)
    OUT.mkdir(exist_ok=True)
    o.to_pickle(OUT / 'orders.pkl')
    q = tape.quotes[['t_us', 'bid', 'ask', 'bid_qty', 'ask_qty']]
    q.to_pickle(OUT / 'quotes.pkl')
    tape.trades.to_pickle(OUT / 'trades.pkl')
    return o


def load():
    p = OUT / 'orders.pkl'
    if not p.exists():
        build()
    return pd.read_pickle(p)


if __name__ == '__main__':
    o = build()
    print(o.head(), len(o))
