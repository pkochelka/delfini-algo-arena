"""The offline replay must see the same trades and top of book as the exchange simulator."""

from __future__ import annotations

import gzip
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis.replay import replay_csv
from explorer.data.orderbook import build_best_quotes
from simulator.simulator import ExchangeSimulator, SimulationConfig
from simulator.strategy import BaseStrategy

SAMPLE_GZ = Path(__file__).resolve().parents[1] / 'data' / 'sample.csv.gz'


class _Idle(BaseStrategy):
    def on_order_book_change(self, snapshot, trades) -> None:  # noqa: ANN001
        pass

    def on_private_order_update(self, event) -> None:  # noqa: ANN001
        pass


@pytest.fixture(scope='module')
def sample_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    tmp = tmp_path_factory.mktemp('parity')
    csv_path = tmp / 'sample.csv'
    with gzip.open(SAMPLE_GZ, 'rb') as src, csv_path.open('wb') as dst:
        shutil.copyfileobj(src, dst)
    config = SimulationConfig(book_delta_log_path=tmp / 'book_deltas.csv', trade_log_path=tmp / 'trades.csv')
    ExchangeSimulator(_Idle(), config).run(csv_path)
    return csv_path, tmp


def test_trades_match_simulator(sample_run: tuple[Path, Path]) -> None:
    csv_path, logs = sample_run
    tape = replay_csv(csv_path)
    sim = pd.read_csv(logs / 'trades.csv')

    assert len(tape.trades) == len(sim)
    np.testing.assert_array_equal(tape.trades['t_us'], sim['timestamp_us'])
    np.testing.assert_array_equal(tape.trades['price'], sim['price'])
    np.testing.assert_array_equal(tape.trades['qty'], sim['quantity'])
    np.testing.assert_array_equal(tape.trades['aggressor'], sim['taker_side'])
    np.testing.assert_array_equal(tape.events['order_id'].to_numpy()[tape.trades['maker_row']], sim['maker_order_id'].astype(str))


def test_best_quotes_match_simulator(sample_run: tuple[Path, Path]) -> None:
    csv_path, logs = sample_run
    tape = replay_csv(csv_path)
    sim = build_best_quotes(pd.read_csv(logs / 'book_deltas.csv'))
    ours = tape.quotes().set_index('t_us').loc[sim['timestamp_us']]

    np.testing.assert_array_equal(ours['bid'].to_numpy(), sim['best_bid'].to_numpy(dtype='float64'))
    np.testing.assert_array_equal(ours['ask'].to_numpy(), sim['best_ask'].to_numpy(dtype='float64'))


def _random_tape(path: Path, n: int = 20_000, seed: int = 7) -> None:
    """Inserts around a drifting mid, a few deep sweeps, IOCs and cancels of live and dead orders."""
    rng = np.random.default_rng(seed)
    rows, live, mid, t = [], [], 10_000, 0.0
    for k in range(n):
        t += rng.exponential(0.01)
        mid += rng.choice([-1, 0, 1], p=[0.05, 0.9, 0.05])
        oid = 1_000_000 + k
        if live and rng.random() < 0.35:
            victim = live.pop(rng.integers(len(live))) if rng.random() < 0.9 else oid - 5
            rows.append(f'{t:.6f},Cancel,{victim},,,,')
            continue
        side = 'B' if rng.random() < 0.5 else 'A'
        aggressive = rng.random() < 0.15
        offset = -int(rng.integers(0, 6)) if aggressive else int(rng.integers(1, 8))
        price = mid - offset if side == 'B' else mid + offset
        vol = int(rng.integers(1, 80 if aggressive else 30))
        lifespan = 'F' if aggressive and rng.random() < 0.5 else 'G'
        rows.append(f'{t:.6f},Insert,{oid},{side},{vol},{price},{lifespan}')
        if lifespan == 'G':
            live.append(oid)
    path.write_text('Time,Operation,OrderId,Side,Volume,Price,Lifespan\n' + '\n'.join(rows) + '\n')


def test_random_tape_parity(tmp_path: Path) -> None:
    csv_path = tmp_path / 'random.csv'
    _random_tape(csv_path)
    config = SimulationConfig(book_delta_log_path=tmp_path / 'deltas.csv', trade_log_path=tmp_path / 'trades.csv')
    ExchangeSimulator(_Idle(), config).run(csv_path)

    tape = replay_csv(csv_path)
    sim = pd.read_csv(tmp_path / 'trades.csv')
    assert (tape.trades['depth'] > 0).any() and tape.orders['ioc'].any()
    assert len(tape.trades) == len(sim)
    np.testing.assert_array_equal(tape.trades[['t_us', 'price', 'qty']].to_numpy(), sim[['timestamp_us', 'price', 'quantity']].to_numpy())
    np.testing.assert_array_equal(tape.trades['aggressor'], sim['taker_side'])

    best = build_best_quotes(pd.read_csv(tmp_path / 'deltas.csv'))
    ours = tape.quotes().set_index('t_us').loc[best['timestamp_us']]
    np.testing.assert_array_equal(ours['bid'].to_numpy(), best['best_bid'].to_numpy(dtype='float64'))
    np.testing.assert_array_equal(ours['ask'].to_numpy(), best['best_ask'].to_numpy(dtype='float64'))
