"""`fast-sim`: `exchange-simulator --no-logs` on the parquet tape cache, same result.json / pnl.csv.

    uv run fast-sim --strategy S.py --backtest data/round3.csv --log-dir logs/x [--set k=v ...] [--jit]

Full logs (run.jsonl, events.csv, ...) are not supported here; use exchange-simulator for those.
"""

from __future__ import annotations

import argparse
import ast
import json
import time
from pathlib import Path

from fastsim.engine import grader_config, run_backtest, write_outputs


def parse_sets(items: list[str]) -> dict:
    """`['k=v', ...]` -> PARAMS overrides, values as Python literals (strings may be left unquoted)."""
    out = {}
    for item in items:
        key, value = item.split('=', 1)
        try:
            out[key] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            out[key] = value
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description='Fast backtest with the grader settings (parquet tape cache)')
    ap.add_argument('--strategy', type=Path, required=True)
    ap.add_argument('--backtest', type=Path, required=True)
    ap.add_argument('--log-dir', type=Path, required=True)
    ap.add_argument('--public-depth', type=int, default=20)
    ap.add_argument('--set', nargs='*', default=[], metavar='K=V', help='PARAMS overrides')
    ap.add_argument('--jit', action='store_true', help="numba-compile the strategy's JIT_KERNELS")
    args = ap.parse_args()

    started = time.perf_counter()
    result, curve = run_backtest(args.strategy, args.backtest, parse_sets(args.set), jit=args.jit,
                                 config=grader_config(args.public_depth))
    write_outputs(args.log_dir, result, curve)
    secs = time.perf_counter() - started
    print(json.dumps({**result.model_dump(mode='json'), 'secs': round(secs, 1), 'log_dir': str(args.log_dir)}))


if __name__ == '__main__':
    main()
