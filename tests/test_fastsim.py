"""fastsim must reproduce the reference simulator exactly: tape columns, results, PnL curve, PARAMS overrides, JIT."""

from __future__ import annotations

import gzip
import shutil
from pathlib import Path

import pytest

from fastsim import tape as tape_mod
from fastsim.engine import FastSimulator, grader_config, load_strategy, run_backtest
from simulator.main import load_strategy_from_file
from simulator.simulator import ExchangeSimulator

SAMPLE_GZ = Path(__file__).resolve().parents[1] / 'data' / 'sample.csv.gz'

# Exercises every engine path: resting limits, modifies, cancels (some racing fills), IOCs, position-limit rejects.
CHURN = '''
import numpy as np
from simulator.models import OrderRequest, OrderType, Side
from simulator.strategy import BaseStrategy

PARAMS = {'qty': 3, 'every': 7, 'ioc_every': 23}
JIT_KERNELS = ('skew',)


def skew(inv, scale):
    return np.floor(inv * scale)


class Churn(BaseStrategy):
    def __init__(self):
        super().__init__()
        self.n = 0
        self.live = []
        self.inv = 0

    def on_private_order_update(self, event):
        if event.filled_quantity:
            self.inv += event.filled_quantity if event.order_id in self.buys else -event.filled_quantity

    def on_order_book_change(self, snapshot, trades):
        self.n += 1
        if not hasattr(self, 'buys'):
            self.buys = set()
        if not snapshot.bids or not snapshot.asks or self.n % PARAMS['every']:
            return
        bb, ba = snapshot.bids[0].price, snapshot.asks[0].price
        k = int(skew(float(self.inv), 0.01))
        if len(self.live) >= 4:
            self.cancel_order(self.live.pop(0))
        if self.live and self.n % 3 == 0:
            self.modify_order(self.live[-1], 1)
        side = Side.BUY if self.n % 2 else Side.SELL
        qty = PARAMS['qty'] * (40 if self.n % 101 == 0 else 1)
        kind = OrderType.IOC if self.n % PARAMS['ioc_every'] == 0 else OrderType.LIMIT
        price = (ba if kind is OrderType.IOC else bb - k) if side is Side.BUY else (bb if kind is OrderType.IOC else ba - k)
        oid = self.send_order(OrderRequest(side=side, price=max(price, 1), quantity=qty, order_type=kind))
        if side is Side.BUY:
            self.buys.add(oid)
        if kind is OrderType.LIMIT:
            self.live.append(oid)
'''


@pytest.fixture(scope='module')
def sample_csv(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp('fastsim') / 'sample.csv'
    with gzip.open(SAMPLE_GZ, 'rb') as src, path.open('wb') as dst:
        shutil.copyfileobj(src, dst)
    return path


@pytest.fixture(scope='module')
def churn_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp('strat') / 'churn.py'
    path.write_text(CHURN)
    return path


def _reference(strategy_file: Path, csv_path: Path, extra_source: str = '') -> tuple[dict, list]:
    if extra_source:
        patched = strategy_file.with_name(f'patched_{strategy_file.name}')
        patched.write_text(strategy_file.read_text() + extra_source)
        strategy_file = patched
    sim = ExchangeSimulator(load_strategy_from_file(strategy_file), grader_config())
    result = sim.run(csv_path)
    points = sim.pnl_curve(result.pnl)
    return result.model_dump(), [((t - points[0][0]) / 1e6, p) for t, p in points]


def test_tape_columns_match_reference_parser(sample_csv: Path) -> None:
    assert tape_mod.verify(sample_csv) == len(tape_mod.load_tape(sample_csv)) > 0


def test_engine_matches_reference(sample_csv: Path, churn_file: Path) -> None:
    ref_result, ref_curve = _reference(churn_file, sample_csv)
    result, curve = run_backtest(churn_file, sample_csv)
    assert ref_result['fills'] > 0 and ref_result['traded_volume'] > 0
    assert result.model_dump() == ref_result
    assert curve == ref_curve


def test_reference_loop_on_columns_matches(sample_csv: Path, churn_file: Path) -> None:
    """With progress on, FastSimulator falls back to the reference run() over the cached columns."""
    ref_result, _ = _reference(churn_file, sample_csv)
    sim = FastSimulator(load_strategy(churn_file), grader_config(show_progress=True, progress_fd=2))
    assert sim.run(sample_csv).model_dump() == ref_result


def test_params_override_equals_appended_update(sample_csv: Path, churn_file: Path) -> None:
    over = {'qty': 2, 'every': 5}
    ref_result, ref_curve = _reference(churn_file, sample_csv, f'\nPARAMS.update({over!r})\n')
    result, curve = run_backtest(churn_file, sample_csv, over)
    assert result.model_dump() == ref_result
    assert curve == ref_curve


def test_jit_kernels_compile_and_match(sample_csv: Path, churn_file: Path) -> None:
    pytest.importorskip('numba')
    strategy = load_strategy(churn_file, jit=True)
    import user_strategy_module as module  # the loader registers the module under this name

    assert type(module.skew).__name__ == 'CPUDispatcher'
    ref_result, _ = _reference(churn_file, sample_csv)
    assert run_backtest(strategy, sample_csv)[0].model_dump() == ref_result
