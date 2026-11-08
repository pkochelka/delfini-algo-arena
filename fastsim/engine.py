"""`FastSimulator`: the reference `ExchangeSimulator` fed from the parquet tape columns.

Matching, latency, position limits, PnL and the strategy callbacks are all inherited unchanged. What
differs is only where the events come from and, when no logs or progress bar are requested, the outer
replay loop: it skips building a `MarketEvent` per row (the tape was validated when it was cached) and
the per-event logging and progress checks. Anything else falls back to the reference `run()`, reading
the same columns through `iter_backtest`.

Parity with `exchange-simulator` (`result.json` and `pnl.csv`, byte for byte) is checked by
`python -m fastsim.parity` and `tests/test_fastsim.py`; re-run them after touching this file or after
`simulator/` changes upstream.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from simulator.models import LiveOrder, MarketEvent, MarketOperation, OrderType, Side, SimulationResult
from simulator.simulator import ExchangeSimulator, SimulationConfig
from simulator.strategy import BaseStrategy

from fastsim.tape import OP_INSERT, SIDE_BUY, Tape, load_tape

# The grader's settings, as in simulator/main.py.
ORDER_ACCEPT_DELAY_MS = 5
MAX_POSITION = 100
MAX_POTENTIAL_POSITION = 1000
POSITION_BREACH_PENALTY_PER_SECOND = 10_000.0
_CHUNK = 1 << 16  # rows converted to Python lists at a time (keeps memory flat)


def grader_config(public_depth: int = 20, **overrides: Any) -> SimulationConfig:
    """The `SimulationConfig` `exchange-simulator --no-logs` uses."""
    return SimulationConfig(
        public_depth=public_depth,
        max_position=MAX_POSITION,
        max_potential_position=MAX_POTENTIAL_POSITION,
        position_breach_penalty_per_second=POSITION_BREACH_PENALTY_PER_SECOND,
        order_accept_delay_ms=ORDER_ACCEPT_DELAY_MS,
        **overrides,
    )


def _chunks(tape: Tape) -> Iterator[tuple[list, ...]]:
    """Yield the tape as Python lists, `_CHUNK` rows at a time; order ids already as their CSV strings."""
    for a in range(0, len(tape), _CHUNK):
        b = a + _CHUNK
        oid = tape.order_id[a:b].tolist()
        if tape.numeric_ids:
            oid = list(map(str, oid))
        yield (tape.t_us[a:b].tolist(), tape.op[a:b].tolist(), oid, tape.side[a:b].tolist(),
               tape.volume[a:b].tolist(), tape.price[a:b].tolist(), tape.ioc[a:b].tolist())


class FastSimulator(ExchangeSimulator):
    """`ExchangeSimulator` reading the cached tape columns; `run(path)` accepts the CSV path as before."""

    def count_backtest_rows(self, path: Path) -> int:  # type: ignore[override]
        return len(load_tape(path))

    def iter_backtest(self, path: Path) -> Iterator[MarketEvent]:  # type: ignore[override]
        buy, sell, limit, ioc_type = Side.BUY, Side.SELL, OrderType.LIMIT, OrderType.IOC
        insert, cancel = MarketOperation.INSERT, MarketOperation.CANCEL
        for t_us, op, oid, side, vol, px, ioc in _chunks(load_tape(path)):
            for i in range(len(t_us)):
                if op[i] == OP_INSERT:
                    yield MarketEvent(timestamp_us=t_us[i], operation=insert, order_id=oid[i],
                                      side=buy if side[i] == SIDE_BUY else sell, volume=vol[i], price=px[i],
                                      order_type=ioc_type if ioc[i] else limit)
                else:
                    yield MarketEvent(timestamp_us=t_us[i], operation=cancel, order_id=oid[i])

    def run(self, backtest_csv: Path) -> SimulationResult:
        cfg = self.config
        if self._log_run or self._log_deltas or self._log_events or self._log_trades or cfg.show_progress:
            return super().run(backtest_csv)
        return self._run_fast(load_tape(backtest_csv))

    def _run_fast(self, tape: Tape) -> SimulationResult:
        """The reference `run()` loop with `_process_market_event` inlined and no logging."""
        buy, sell, limit, ioc_type = Side.BUY, Side.SELL, OrderType.LIMIT, OrderType.IOC
        activate = self._activate_pending_orders_up_to
        advance = self._advance_time
        execute_taker = self._execute_taker
        rest = self._rest
        remove = self._remove
        emit = self._emit_public_snapshot
        for t_us, op, oid, side, vol, px, ioc in _chunks(tape):
            for i in range(len(t_us)):
                ts = t_us[i]
                self._last_event_us = ts
                if self._pending:
                    activate(ts)
                advance(ts)
                if op[i] == OP_INSERT:
                    order = LiveOrder(order_id=oid[i], owner='market', side=buy if side[i] == SIDE_BUY else sell,
                                      price=px[i], initial_quantity=vol[i], remaining_quantity=vol[i],
                                      order_type=ioc_type if ioc[i] else limit, created_at_us=ts)
                    execute_taker(order)
                    if order.order_type is limit and order.remaining_quantity > 0:
                        rest(order)
                else:
                    remove(oid[i])
                emit()

        activate(None)
        return SimulationResult(
            fills=len(self._filled_order_ids),
            traded_volume=self._user_traded_volume,
            inventory=self._inventory,
            cash=self._cash,
            mark_price=self._mark_price(),
            breach_penalty=self._breach_penalty,
            pnl=self._mark_to_market_pnl(),
            pnl_se=self._pnl_standard_error(),
            abs_inventory_q50=self._abs_inventory_quantile(0.5),
            abs_inventory_q68=self._abs_inventory_quantile(0.68),
            abs_inventory_q95=self._abs_inventory_quantile(0.95),
            abs_inventory_q997=self._abs_inventory_quantile(0.997),
        )


def load_strategy(path: str | Path, params: Mapping[str, Any] | None = None, *, jit: bool = False) -> BaseStrategy:
    """Import a strategy file and return its `BaseStrategy` subclass instance, as `exchange-simulator` does.

    `params` updates the module's `PARAMS` dict after the module has run, which is what appending
    `PARAMS.update(params)` to the source did. With `jit`, every function named in the module's
    `JIT_KERNELS` tuple is replaced by `numba.njit(cache=True)` of itself before the strategy is built;
    the file itself never imports numba, so it is submitted unchanged.
    """
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location('user_strategy_module', path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Failed to load module spec from {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules['user_strategy_module'] = module
    spec.loader.exec_module(module)
    namespace = vars(module)

    if params:
        if not isinstance(namespace.get('PARAMS'), dict):
            raise ValueError(f'{path.name} has no PARAMS dict to apply {dict(params)} to')
        namespace['PARAMS'].update(params)
    if jit:
        kernels = namespace.get('JIT_KERNELS', ())
        if kernels:
            import numba

            for name in kernels:
                namespace[name] = numba.njit(cache=True)(namespace[name])

    for name in sorted(namespace):
        candidate = namespace[name]
        if isinstance(candidate, type) and issubclass(candidate, BaseStrategy) and candidate is not BaseStrategy:
            return candidate()
    raise ValueError(f'No BaseStrategy subclass found in {path}')


def run_backtest(
    strategy: str | Path | BaseStrategy,
    backtest: str | Path,
    params: Mapping[str, Any] | None = None,
    *,
    jit: bool = False,
    config: SimulationConfig | None = None,
) -> tuple[SimulationResult, list[tuple[float, float]]]:
    """One backtest with the grader's settings; returns the result and the PnL curve as (elapsed_s, pnl)."""
    if not isinstance(strategy, BaseStrategy):
        strategy = load_strategy(strategy, params, jit=jit)
    sim = FastSimulator(strategy, config or grader_config())
    result = sim.run(Path(backtest))
    points = sim.pnl_curve(result.pnl)
    start_us = points[0][0]
    return result, [((ts - start_us) / 1_000_000, pnl) for ts, pnl in points]


def write_outputs(log_dir: Path, result: SimulationResult, curve: list[tuple[float, float]]) -> None:
    """Write `result.json` and `pnl.csv` exactly as `exchange-simulator` does."""
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / 'result.json', 'w') as file:
        json.dump(result.model_dump(mode='json'), file, indent=2)
    with open(log_dir / 'pnl.csv', 'w', newline='') as file:
        writer = csv.writer(file, lineterminator='\n')
        writer.writerow(['elapsed_s', 'pnl'])
        writer.writerows(curve)
