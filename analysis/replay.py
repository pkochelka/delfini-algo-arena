"""Replay a raw market tape through a price-time order book, the way the simulator does.

The tape (`Time,Operation,OrderId,Side,Volume,Price,Lifespan`) has no trades in
it: they happen when an insert crosses the opposite side. This module rebuilds
the book event by event with the simulator's matching rules (see
`ExchangeSimulator._execute_taker`: best price first, FIFO within a level, fill
at the maker's price, an IOC (`F`) residual is dropped, a limit (`G`) residual
rests) and records:

- `events`: one row per tape row, with the top of book *after* that row.
- `trades`: every inferred fill, with the aggressor side and how many ticks
  past the touch it reached.
- `orders`: one row per insert, covering its life: where it was placed relative
  to the touch, how much filled, and when and how it ended.
- depth snapshots: the top `DEPTH_LEVELS` price levels per side on a fixed time
  grid (the book as of each grid time), for depth profiles and heatmaps.

Prices are integer ticks. An empty side's price is recorded as 0 and turned into
NaN by the `Tape` accessors.
"""

from __future__ import annotations

from array import array
from bisect import insort
from collections import deque
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

REPLAY_VERSION = 3  # queue ahead / resting quantities added to order records
DEPTH_LEVELS = 20
DEPTH_GRID_US = 100_000

BID, ASK = 0, 1
# orders.end_reason codes
ALIVE, CANCELLED, FILLED, EXPIRED = 0, 1, 2, 3
END_REASONS = {ALIVE: 'alive', CANCELLED: 'cancelled', FILLED: 'filled', EXPIRED: 'ioc_expired'}


@dataclass
class Tape:
    """Everything the replay produced for one CSV."""

    name: str
    events: pd.DataFrame
    trades: pd.DataFrame
    orders: pd.DataFrame
    depth_t_us: np.ndarray  # (S,)
    depth_bid_px: np.ndarray  # (S, DEPTH_LEVELS), best first, 0 where no level
    depth_bid_qty: np.ndarray
    depth_ask_px: np.ndarray
    depth_ask_qty: np.ndarray

    def __getstate__(self) -> dict:
        state = dict(self.__dict__)
        state.pop('quotes', None)  # derived; recomputed on demand
        return state

    @property
    def start_us(self) -> int:
        """The round clock's zero: the first event's timestamp. Every `t_s` column counts from it."""
        return int(self.events['t_us'].iloc[0])

    @property
    def end_us(self) -> int:
        return int(self.events['t_us'].iloc[-1])

    @property
    def duration_s(self) -> float:
        return (self.end_us - self.start_us) / 1e6

    @cached_property
    def quotes(self) -> pd.DataFrame:
        """Top of book after each event, one row per distinct timestamp (the last state at it).

        Columns: t_us, t_s (seconds since the first event), bid, ask, bid_qty,
        ask_qty, bid_n, ask_n, mid, spread, microprice, imbalance.
        """
        ev = self.events
        last_at_ts = ~ev['t_us'].duplicated(keep='last').to_numpy()
        q = ev.loc[last_at_ts, ['t_us', 'bid', 'ask', 'bid_qty', 'ask_qty', 'bid_n', 'ask_n']].reset_index(drop=True)
        q['bid'] = q['bid'].where(q['bid'] > 0).astype('float64')
        q['ask'] = q['ask'].where(q['ask'] > 0).astype('float64')
        q['t_s'] = (q['t_us'] - self.start_us) / 1e6
        q['mid'] = (q['bid'] + q['ask']) / 2
        q['spread'] = q['ask'] - q['bid']
        total = (q['bid_qty'] + q['ask_qty']).astype('float64')
        both_sides = q['bid'].notna() & q['ask'].notna()
        q['imbalance'] = (q['bid_qty'] / total).where(both_sides)
        q['microprice'] = q['ask'] * q['imbalance'] + q['bid'] * (1 - q['imbalance'])
        return q


def read_tape_csv(path: Path) -> pd.DataFrame:
    """Read a tape CSV into typed columns; OrderIds are kept as strings, like the simulator does."""
    df = pd.read_csv(
        path,
        dtype={'Operation': 'string', 'OrderId': 'string', 'Side': 'string', 'Lifespan': 'string'},
    )
    df['t_us'] = np.rint(df['Time'].to_numpy(dtype='float64') * 1e6).astype('int64')
    return df


def replay_csv(path: Path, depth_grid_us: int = DEPTH_GRID_US, depth_levels: int = DEPTH_LEVELS) -> Tape:
    """Replay the tape at `path`. Pure Python, about 1-3 s per million rows; use `analysis.load_tape` to cache."""
    raw = read_tape_csv(path)
    n = len(raw)

    t_all = raw['t_us'].to_numpy().tolist()
    is_insert = (raw['Operation'] == 'Insert').to_numpy(dtype=bool, na_value=False).tolist()
    oid_codes, _ = pd.factorize(raw['OrderId'])
    oids = oid_codes.tolist()
    sides = (raw['Side'] == 'A').to_numpy(dtype=bool, na_value=False).astype(np.int8).tolist()
    vols = raw['Volume'].fillna(0).to_numpy(dtype='int64').tolist()
    pxs = raw['Price'].fillna(0).to_numpy(dtype='int64').tolist()
    ioc = (raw['Lifespan'] == 'F').to_numpy(dtype=bool, na_value=False).tolist()

    # Book state. levels[side][price] -> deque of [oid, remaining, insert_row]
    levels: tuple[dict, dict] = ({}, {})
    level_qty: tuple[dict, dict] = ({}, {})
    prices: tuple[list, list] = ([], [])  # sorted ascending
    live: dict[int, tuple[int, int, list]] = {}  # oid -> (side, price, entry)

    # Per-event top of book (after the event)
    e_bid, e_ask = array('q'), array('q')
    e_bq, e_aq = array('q'), array('q')
    e_bn, e_an = array('l'), array('l')

    # Trades
    tr_t, tr_px, tr_qty = array('q'), array('q'), array('q')
    tr_side = array('b')  # aggressor: 0 buy, 1 sell
    tr_taker, tr_maker = array('q'), array('q')  # insert row of taker / maker
    tr_depth = array('q')  # ticks past the touch at the taker's arrival

    # Orders, indexed by insert row; filled lazily
    o_filled = np.zeros(n, dtype=np.int64)
    o_end_t = np.full(n, -1, dtype=np.int64)
    o_end = np.zeros(n, dtype=np.int8)
    o_dist = np.full(n, np.nan)  # ticks inside (+) / behind (-) the same-side touch at insert
    o_cross = np.zeros(n, dtype=bool)
    o_ahead = np.full(n, -1, dtype=np.int64)
    o_resting = np.zeros(n, dtype=np.int64)

    # Depth snapshots
    d_t: list[int] = []
    d_rows: list[tuple[list, list, list, list]] = []
    next_grid = (t_all[0] // depth_grid_us + 1) * depth_grid_us if n else 0

    def snapshot() -> tuple[list, list, list, list]:
        bp = prices[BID][-depth_levels:][::-1]
        ap = prices[ASK][:depth_levels]
        return bp, [level_qty[BID][p] for p in bp], ap, [level_qty[ASK][p] for p in ap]

    for i in range(n):
        t = t_all[i]
        while t >= next_grid:
            d_t.append(next_grid)
            d_rows.append(snapshot())
            next_grid += depth_grid_us

        if is_insert[i]:
            s = sides[i]
            p = pxs[i]
            v = vols[i]
            o = oids[i]
            opp = 1 - s
            opp_prices = prices[opp]
            same_prices = prices[s]
            if same_prices:
                o_dist[i] = p - same_prices[-1] if s == BID else same_prices[0] - p
            if opp_prices:
                touch = opp_prices[0] if opp == ASK else opp_prices[-1]
                if (p >= touch) if s == BID else (p <= touch):
                    o_cross[i] = True
                    opp_levels = levels[opp]
                    opp_qty = level_qty[opp]
                    while v > 0 and opp_prices:
                        best = opp_prices[0] if opp == ASK else opp_prices[-1]
                        if (p < best) if s == BID else (p > best):
                            break
                        queue = opp_levels[best]
                        while v > 0 and queue:
                            maker = queue[0]
                            q = v if v < maker[1] else maker[1]
                            v -= q
                            maker[1] -= q
                            opp_qty[best] -= q
                            mrow = maker[2]
                            o_filled[mrow] += q
                            o_filled[i] += q
                            tr_t.append(t)
                            tr_px.append(best)
                            tr_qty.append(q)
                            tr_side.append(s)
                            tr_taker.append(i)
                            tr_maker.append(mrow)
                            tr_depth.append(best - touch if s == BID else touch - best)
                            if maker[1] == 0:
                                queue.popleft()
                                del live[maker[0]]
                                o_end[mrow] = FILLED
                                o_end_t[mrow] = t
                        if not queue:
                            del opp_levels[best]
                            del opp_qty[best]
                            if opp == ASK:
                                opp_prices.pop(0)
                            else:
                                opp_prices.pop()
            if v == 0:
                o_end[i] = FILLED
                o_end_t[i] = t
            elif ioc[i]:
                o_end[i] = EXPIRED
                o_end_t[i] = t
            else:
                lv = levels[s]
                o_ahead[i] = level_qty[s].get(p, 0)
                o_resting[i] = v
                if p not in lv:
                    lv[p] = deque()
                    level_qty[s][p] = 0
                    insort(same_prices, p)
                entry = [o, v, i]
                lv[p].append(entry)
                level_qty[s][p] += v
                live[o] = (s, p, entry)
        else:
            found = live.pop(oids[i], None)
            if found is not None:
                s, p, entry = found
                queue = levels[s][p]
                queue.remove(entry)
                level_qty[s][p] -= entry[1]
                o_end[entry[2]] = CANCELLED
                o_end_t[entry[2]] = t
                if not queue:
                    del levels[s][p]
                    del level_qty[s][p]
                    prices[s].remove(p)

        bp = prices[BID]
        ap = prices[ASK]
        if bp:
            b = bp[-1]
            e_bid.append(b)
            e_bq.append(level_qty[BID][b])
            e_bn.append(len(levels[BID][b]))
        else:
            e_bid.append(0)
            e_bq.append(0)
            e_bn.append(0)
        if ap:
            a = ap[0]
            e_ask.append(a)
            e_aq.append(level_qty[ASK][a])
            e_an.append(len(levels[ASK][a]))
        else:
            e_ask.append(0)
            e_aq.append(0)
            e_an.append(0)

    events = pd.DataFrame(
        {
            't_us': raw['t_us'].to_numpy(),
            'op': np.where(np.array(is_insert, dtype=bool), 'I', 'C'),
            'order_id': raw['OrderId'].to_numpy(),
            'side': raw['Side'].to_numpy(),
            'volume': raw['Volume'].to_numpy(),
            'price': raw['Price'].to_numpy(),
            'lifespan': raw['Lifespan'].to_numpy(),
            'bid': np.frombuffer(e_bid, dtype=np.int64),
            'ask': np.frombuffer(e_ask, dtype=np.int64),
            'bid_qty': np.frombuffer(e_bq, dtype=np.int64),
            'ask_qty': np.frombuffer(e_aq, dtype=np.int64),
            'bid_n': np.asarray(e_bn, dtype=np.int32),
            'ask_n': np.asarray(e_an, dtype=np.int32),
        }
    )
    for col in ('op', 'side', 'lifespan'):
        events[col] = events[col].astype('category')

    trades = pd.DataFrame(
        {
            't_us': np.frombuffer(tr_t, dtype=np.int64),
            'price': np.frombuffer(tr_px, dtype=np.int64),
            'qty': np.frombuffer(tr_qty, dtype=np.int64),
            'aggressor': np.where(np.frombuffer(tr_side, dtype=np.int8) == BID, 'B', 'A'),
            'taker_row': np.frombuffer(tr_taker, dtype=np.int64),
            'maker_row': np.frombuffer(tr_maker, dtype=np.int64),
            'depth': np.frombuffer(tr_depth, dtype=np.int64),
        }
    )

    ins = np.flatnonzero(np.array(is_insert, dtype=bool))
    orders = pd.DataFrame(
        {
            'row': ins,
            't_us': raw['t_us'].to_numpy()[ins],
            'order_id': raw['OrderId'].to_numpy()[ins],
            'side': raw['Side'].to_numpy()[ins],
            'price': raw['Price'].to_numpy()[ins].astype('int64'),
            'volume': raw['Volume'].to_numpy()[ins].astype('int64'),
            'ioc': np.array(ioc, dtype=bool)[ins],
            'dist': o_dist[ins],
            'crossed': o_cross[ins],
            'queue_ahead': o_ahead[ins],
            'resting_quantity': o_resting[ins],
            'filled': o_filled[ins],
            'end_t_us': o_end_t[ins],
            'end': pd.Categorical.from_codes(o_end[ins], categories=list(END_REASONS.values())),
        }
    )
    orders['end_t_us'] = orders['end_t_us'].where(orders['end_t_us'] >= 0)

    s_count = len(d_t)
    dbp = np.zeros((s_count, depth_levels), dtype=np.int64)
    dbq = np.zeros_like(dbp)
    dap = np.zeros_like(dbp)
    daq = np.zeros_like(dbp)
    for k, (bp, bq, ap, aq) in enumerate(d_rows):
        dbp[k, : len(bp)] = bp
        dbq[k, : len(bq)] = bq
        dap[k, : len(ap)] = ap
        daq[k, : len(aq)] = aq

    return Tape(
        name=path.stem,
        events=events,
        trades=trades,
        orders=orders,
        depth_t_us=np.asarray(d_t, dtype=np.int64),
        depth_bid_px=dbp,
        depth_bid_qty=dbq,
        depth_ask_px=dap,
        depth_ask_qty=daq,
    )
