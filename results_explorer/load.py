"""Read one simulator run directory: result, metadata, PnL curve and, for full-logs runs, the user's orders and fills.

Nothing here touches the simulator. Inventory, cash and order lifetimes are
reconstructed from trades.csv / events.csv. The market mid comes from
`analysis.load_tape` (the cached replay of the tape the run used). That replay
is the market-only book, which is also what the simulator marks inventory against.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_TAPES: dict[Path, Any] = {}


@dataclass
class Run:
    """One run directory, loaded."""

    name: str
    log_dir: Path
    result: dict[str, Any]
    metadata: dict[str, Any]
    pnl: pd.DataFrame  # elapsed_s, pnl (the simulator's, sampled every second)
    start_us: int  # the round clock's zero: pnl.csv's elapsed_s = (t_us - start_us) / 1e6
    end_us: int
    # Full-logs runs only (None after --no-logs):
    fills: pd.DataFrame | None = None  # t_us, price, qty, side (+1 user buys / -1 sells), maker (bool), order_id
    orders: pd.DataFrame | None = None  # order_id, side, price, qty, sent_us, accept_us, end_us, filled, outcome
    order_counts: dict[str, int] = field(default_factory=dict)
    # Market top of book (from the tape replay), None if the tape is unavailable:
    quotes: pd.DataFrame | None = None  # t_us, bid, ask, mid
    warnings: list[str] = field(default_factory=list)

    @property
    def full_logs(self) -> bool:
        return self.fills is not None

    @property
    def position_limit(self) -> int:
        return int(self.metadata.get('position_limit', 100))

    @property
    def duration_s(self) -> float:
        return (self.end_us - self.start_us) / 1e6


def _tape_path(log_dir: Path, metadata: dict[str, Any]) -> Path | None:
    candidates = [metadata.get('data_file')]
    run_log = log_dir / 'run.jsonl'
    if run_log.exists():
        with run_log.open() as f:
            first = json.loads(f.readline())
        candidates.append(first.get('payload', {}).get('backtest_csv'))
    for c in candidates:
        if c and Path(c).exists():
            return Path(c).resolve()
    return None


def _load_quotes(path: Path) -> pd.DataFrame:
    if path not in _TAPES:
        from analysis import load_tape

        tape = load_tape(path)
        q = tape.quotes[['t_us', 'bid', 'ask', 'mid']].copy()
        q['mid'] = q['mid'].ffill().bfill()
        _TAPES[path] = (q, tape.start_us, tape.end_us)
    return _TAPES[path]


def _load_fills(log_dir: Path) -> pd.DataFrame:
    tr = pd.read_csv(
        log_dir / 'trades.csv',
        dtype={'timestamp_us': 'int64', 'taker_order_id': 'str', 'maker_order_id': 'str'},
    )
    tr = tr[(tr['taker_owner'] == 'user') | (tr['maker_owner'] == 'user')]
    maker = (tr['maker_owner'] == 'user').to_numpy()
    taker_buys = (tr['taker_side'] == 'B').to_numpy()
    # As taker the user trades on taker_side; as maker on the opposite side (a selling taker hits the user's bid).
    side = np.where(maker, np.where(taker_buys, -1, 1), np.where(taker_buys, 1, -1))
    return pd.DataFrame(
        {
            't_us': tr['timestamp_us'].to_numpy(),
            'price': tr['price'].to_numpy(dtype='float64'),
            'qty': tr['quantity'].to_numpy(dtype='int64'),
            'side': side.astype('int64'),
            'maker': maker,
            'order_id': np.where(maker, tr['maker_order_id'], tr['taker_order_id']),
        }
    ).reset_index(drop=True)


def _load_orders(log_dir: Path, fills: pd.DataFrame, end_us: int) -> tuple[pd.DataFrame, dict[str, int]]:
    """The user's order lifecycle from events.csv (only its user rows are kept)."""
    ev = pd.read_csv(
        log_dir / 'events.csv',
        dtype={'timestamp_us': 'int64', 'order_id': 'str', 'source': 'str', 'operation': 'str'},
    )
    ev = ev[ev['source'] == 'user']
    counts = {str(k): int(v) for k, v in ev['operation'].value_counts().items()}

    sent = ev[ev['operation'] == 'add_sent'].set_index('order_id')
    acc = ev[ev['operation'] == 'add'].drop_duplicates('order_id').set_index('order_id')
    canc = ev[ev['operation'] == 'cancel'].drop_duplicates('order_id').set_index('order_id')['timestamp_us']
    failed = set(ev.loc[ev['operation'] == 'add_failed', 'order_id'])

    orders = pd.DataFrame(
        {
            'side': sent['side'].map({'B': 1, 'A': -1}),
            'price': sent['price'],
            'qty': sent['quantity'],
            'sent_us': sent['timestamp_us'],
        }
    )
    orders['accept_us'] = acc['timestamp_us'].reindex(orders.index)
    g = fills.groupby('order_id')
    orders['filled'] = g['qty'].sum().reindex(orders.index).fillna(0).astype('int64')
    orders['last_fill_us'] = g['t_us'].max().reindex(orders.index)
    orders['first_fill_us'] = g['t_us'].min().reindex(orders.index)
    orders['taker'] = g['maker'].min().reindex(orders.index).eq(False)
    orders['cancel_us'] = canc.reindex(orders.index)

    full = orders['filled'] >= orders['qty']
    cancelled = orders['cancel_us'].notna()
    outcome = np.select(
        [orders.index.isin(failed), full, cancelled & (orders['filled'] > 0), cancelled],
        ['rejected', 'filled', 'partial+cancelled', 'cancelled'],
        default='alive at end',
    )
    orders['outcome'] = outcome
    # IOC residuals are cancelled at activation, which counts as 'cancelled' (or 'partial+cancelled') here too.
    end = np.where(full, orders['last_fill_us'], np.where(cancelled, orders['cancel_us'], end_us))
    orders['end_us'] = pd.Series(end, index=orders.index).where(orders['accept_us'].notna())
    orders.index.name = 'order_id'
    return orders.reset_index(), counts


def load_run(log_dir: str | Path, name: str | None = None) -> Run:
    log_dir = Path(log_dir)
    result = json.loads((log_dir / 'result.json').read_text())
    meta_path = log_dir / 'metadata.json'
    metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    pnl = pd.read_csv(log_dir / 'pnl.csv')
    warnings: list[str] = []

    quotes = None
    tape = _tape_path(log_dir, metadata)
    if tape is not None:
        quotes, start_us, end_us = _load_quotes(tape)
    else:
        warnings.append('tape CSV not found: no market mid, so no markouts, edge decomposition or price panel')
        start_us, end_us = 0, int(pnl['elapsed_s'].iloc[-1] * 1e6)

    run = Run(
        name=name or log_dir.name,
        log_dir=log_dir,
        result=result,
        metadata=metadata,
        pnl=pnl,
        start_us=start_us,
        end_us=end_us,
        quotes=quotes,
        warnings=warnings,
    )
    if (log_dir / 'trades.csv').exists() and (log_dir / 'events.csv').exists():
        run.fills = _load_fills(log_dir)
        if tape is None:
            run.start_us = int(pd.read_csv(log_dir / 'events.csv', nrows=1)['timestamp_us'].iloc[0])
        run.orders, run.order_counts = _load_orders(log_dir, run.fills, run.end_us)
    else:
        warnings.append('run was made with --no-logs: only the PnL curve and result.json are available')
    return run
