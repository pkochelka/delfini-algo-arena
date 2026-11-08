"""Time each stage of a backtest, reference vs fast engine, one run at a time.

    uv run python -m fastsim.bench --backtest data/round3.csv [--strategy S.py ...] [--jit]

Stages: CSV parse (reference parser) vs parquet load, the engine with an idle strategy, then each
--strategy on both engines. Wall time depends on machine load (4 performance + 6 efficiency cores), so CPU time is printed too
and --repeat alternates reference and fast runs so both see the same conditions.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from simulator.main import load_strategy_from_file
from simulator.simulator import ExchangeSimulator
from simulator.strategy import BaseStrategy

from fastsim import tape as tape_mod
from fastsim.engine import FastSimulator, grader_config, load_strategy


def _timed(label: str, fn) -> float:
    """Print wall and process-CPU seconds; CPU time is the fairer number on a busy machine."""
    started, cpu = time.perf_counter(), time.process_time()
    out = fn()
    secs, cpu = time.perf_counter() - started, time.process_time() - cpu
    extra = f'  pnl {out.pnl:,.2f}' if hasattr(out, 'pnl') else ''
    print(f'{label:52s} wall {secs:7.2f}s  cpu {cpu:7.2f}s{extra}', flush=True)
    return cpu


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--backtest', type=Path, required=True)
    ap.add_argument('--strategy', type=Path, nargs='*', default=[])
    ap.add_argument('--jit', action='store_true', help='also time the fast engine with JIT_KERNELS compiled')
    ap.add_argument('--skip-reference', action='store_true', help='only time the fast engine')
    ap.add_argument('--repeat', type=int, default=1, help='alternate reference/fast runs this many times')
    args = ap.parse_args()
    path = args.backtest
    tape_mod.ensure_cache(path)

    if not args.skip_reference:
        _timed('parse CSV (reference)', lambda: sum(1 for _ in ExchangeSimulator.iter_backtest(path)))
    _timed('load parquet (cold, this process)', lambda: tape_mod.load_tape(path))
    if not args.skip_reference:
        _timed('engine, idle strategy (reference)', lambda: ExchangeSimulator(BaseStrategy(), grader_config()).run(path))
    _timed('engine, idle strategy (fast)', lambda: FastSimulator(BaseStrategy(), grader_config()).run(path))
    for strat in [s for s in args.strategy for _ in range(args.repeat)]:
        if not args.skip_reference:
            _timed(f'{strat.name} (reference)',
                   lambda: ExchangeSimulator(load_strategy_from_file(strat), grader_config()).run(path))
        _timed(f'{strat.name} (fast)', lambda: FastSimulator(load_strategy(strat), grader_config()).run(path))
        if args.jit:
            _timed(f'{strat.name} (fast, jit; includes compile)',
                   lambda: FastSimulator(load_strategy(strat, jit=True), grader_config()).run(path))


if __name__ == '__main__':
    main()
