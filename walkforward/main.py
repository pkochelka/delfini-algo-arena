"""CLI: walk-forward evaluation of one strategy over consecutive out-of-sample windows of a backtest.

    uv run walk-forward --strategy strategies/trend_follower.py --backtest data/round1.csv --folds 10

The tape is cut into `--folds` consecutive eval windows (optionally only the first `--end-frac` of
it, keeping the rest held out). Each fold is an independent, fresh replay: a new strategy instance on
a file that starts `--warmup-s` before its eval window with the resting book seeded, so the strategy
can learn on the warmup before it is scored. A fold's score is the PnL earned inside its eval window,
PnL(end) - PnL(eval_start): a strategy whose PARAMS has a `trade_start_us` key is told not to trade
before eval_start; any other strategy trades through the warmup and carries that inventory in.

Writes to --out-dir (default logs/wf/<strategy>-<tape>): config.json, folds.csv (one row per fold),
summary.json (aggregate metrics), pnl/<fold>.csv (eval-window PnL curve) and report.html.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from walkforward.folds import Fold, make_folds

FOLD_FIELDS = ['fold', 'start_s', 'eval_start_s', 'end_s', 'eval_pnl', 'warmup_pnl', 'max_drawdown',
               'fills', 'traded_volume', 'end_inventory', 'breach_penalty', 'pnl_se', 'abs_inventory_q50',
               'abs_inventory_q95', 'abs_inventory_q997', 'rows', 'runtime_s']


def load_strategy(path: Path, overrides: dict, trade_start_us: int):
    """Import a private copy of the strategy file, apply PARAMS overrides, return a fresh instance."""
    from simulator.strategy import BaseStrategy

    spec = importlib.util.spec_from_file_location(f'wf_strategy_{trade_start_us}', path.resolve())
    if spec is None or spec.loader is None:
        raise ImportError(f'cannot load {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look their module up here
    sys.path.insert(0, str(path.resolve().parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    params = getattr(module, 'PARAMS', None)
    if isinstance(params, dict):
        params.update(overrides)
        if 'trade_start_us' in params:
            params['trade_start_us'] = trade_start_us
    elif overrides:
        raise ValueError(f'--params given but {path.name} has no PARAMS dict')
    for name in sorted(dir(module)):
        cand = getattr(module, name)
        if isinstance(cand, type) and issubclass(cand, BaseStrategy) and cand is not BaseStrategy \
                and cand.__module__ == module.__name__:
            return cand()
    raise ValueError(f'no BaseStrategy subclass in {path}')


def run_fold(strategy_path: str, overrides: dict, fold: Fold) -> dict:
    """Worker: replay one fold; return its metrics and its eval-window PnL curve."""
    from simulator.main import (MAX_POSITION, MAX_POTENTIAL_POSITION, ORDER_ACCEPT_DELAY_MS,
                                POSITION_BREACH_PENALTY_PER_SECOND)
    from simulator.simulator import ExchangeSimulator, SimulationConfig

    t = time.perf_counter()
    strategy = load_strategy(Path(strategy_path), overrides, fold.eval_start_us)
    config = SimulationConfig(max_position=MAX_POSITION, max_potential_position=MAX_POTENTIAL_POSITION,
                              position_breach_penalty_per_second=POSITION_BREACH_PENALTY_PER_SECOND,
                              order_accept_delay_ms=ORDER_ACCEPT_DELAY_MS)
    sim = ExchangeSimulator(strategy=strategy, config=config)
    r = sim.run(Path(fold.path))
    points = sim.pnl_curve(r.pnl)

    base = 0.0  # PnL already made (or lost) when the eval window opens
    for ts, pnl in points:
        if ts > fold.eval_start_us:
            break
        base = pnl
    curve = [(fold.eval_start_s, 0.0)] + [(ts / 1e6, pnl - base) for ts, pnl in points if ts > fold.eval_start_us]
    peak, mdd = 0.0, 0.0
    for _, v in curve:
        peak = max(peak, v)
        mdd = max(mdd, peak - v)
    return {
        'fold': fold.name, 'start_s': fold.start_s, 'eval_start_s': fold.eval_start_s, 'end_s': fold.end_s,
        'eval_pnl': r.pnl - base, 'warmup_pnl': base, 'max_drawdown': mdd, 'fills': r.fills,
        'traded_volume': r.traded_volume, 'end_inventory': r.inventory, 'breach_penalty': r.breach_penalty,
        'pnl_se': r.pnl_se, 'abs_inventory_q50': r.abs_inventory_q50, 'abs_inventory_q95': r.abs_inventory_q95,
        'abs_inventory_q997': r.abs_inventory_q997, 'rows': fold.rows,
        'runtime_s': time.perf_counter() - t, 'curve': curve,
    }


def summarize(rows: list[dict]) -> dict:
    """Aggregate fold metrics; the stitched curve chains the eval windows end to end."""
    pnls = [r['eval_pnl'] for r in rows]
    n = len(pnls)
    mean = statistics.fmean(pnls)
    sd = statistics.stdev(pnls) if n > 1 else 0.0
    offset, peak, mdd = 0.0, 0.0, 0.0
    for r in rows:
        for _, v in r['curve']:
            peak = max(peak, offset + v)
            mdd = max(mdd, peak - offset - v)
        offset += r['eval_pnl']
    hours = sum(r['end_s'] - r['eval_start_s'] for r in rows) / 3600
    return {
        'folds': n,
        'total_pnl': sum(pnls),
        'mean_pnl': mean,
        'median_pnl': statistics.median(pnls),
        'std_pnl': sd,
        'min_pnl': min(pnls),
        'max_pnl': max(pnls),
        'win_rate': sum(p > 0 for p in pnls) / n,
        'fold_sharpe': mean / sd if sd else None,  # mean / std of fold PnL, not annualised
        't_stat': mean / (sd / math.sqrt(n)) if sd else None,
        'pnl_per_hour': sum(pnls) / hours if hours else None,
        'stitched_max_drawdown': mdd,
        'worst_fold_drawdown': max(r['max_drawdown'] for r in rows),
        'total_fills': sum(r['fills'] for r in rows),
        'total_volume': sum(r['traded_volume'] for r in rows),
        'total_breach_penalty': sum(r['breach_penalty'] for r in rows),
        'mean_abs_inventory_q95': statistics.fmean(r['abs_inventory_q95'] for r in rows),
        'max_abs_inventory_q997': max(r['abs_inventory_q997'] for r in rows),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--strategy', type=Path, required=True, help='strategy .py file (a BaseStrategy subclass)')
    p.add_argument('--backtest', type=Path, required=True, help='backtest CSV')
    p.add_argument('--folds', type=int, default=10, help='number of consecutive eval windows (default 10)')
    p.add_argument('--warmup-s', type=float, default=0.0, help='seconds of tape before each eval window (default 0)')
    p.add_argument('--end-frac', type=float, default=1.0,
                   help='only use the first fraction of the tape; the rest stays held out (default 1.0)')
    p.add_argument('--params', type=Path, help='JSON of PARAMS overrides (flat, or {"params": {...}} as tune.py writes)')
    p.add_argument('--out-dir', type=Path, help='default logs/wf/<strategy>-<tape>')
    p.add_argument('--jobs', type=int, default=None, help='parallel fold replays (default: CPU count)')
    p.add_argument('--no-report', action='store_true', help='skip writing report.html')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 < args.end_frac <= 1:
        sys.exit('--end-frac must be in (0, 1]')
    overrides: dict = {}
    if args.params:
        overrides = json.loads(args.params.read_text())
        overrides = overrides.get('params', overrides)
    out_dir: Path = args.out_dir or Path('logs/wf') / f'{args.strategy.stem}-{args.backtest.stem}'
    (out_dir / 'pnl').mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    folds = make_folds(args.backtest, args.folds, args.warmup_s, args.end_frac)
    print(f'{len(folds)} folds ready ({time.perf_counter() - t0:.1f}s); replaying...', flush=True)

    rows: list[dict] = []
    jobs = min(args.jobs or os.cpu_count() or 1, len(folds))
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(run_fold, str(args.strategy), overrides, f) for f in folds]
        for fut in as_completed(futures):
            r = fut.result()
            rows.append(r)
            print(f'  {r["fold"]}: eval {r["eval_start_s"]:8.1f}s -> {r["end_s"]:8.1f}s  pnl {r["eval_pnl"]:+11.2f}'
                  f'  fills {r["fills"]:6d}  q95 {r["abs_inventory_q95"]:4d}  ({r["runtime_s"]:.0f}s)', flush=True)
    rows.sort(key=lambda r: r['eval_start_s'])

    summary = summarize(rows)
    config = {
        'strategy': str(args.strategy.resolve()), 'strategy_name': args.strategy.stem,
        'backtest': str(args.backtest.resolve()), 'folds': args.folds, 'warmup_s': args.warmup_s,
        'end_frac': args.end_frac, 'params': overrides, 'run_timestamp': datetime.now().isoformat(timespec='seconds'),
        'wall_time_s': time.perf_counter() - t0, 'fold_files': [asdict(f) for f in folds],
    }
    (out_dir / 'config.json').write_text(json.dumps(config, indent=2))
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
    with (out_dir / 'folds.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, FOLD_FIELDS, extrasaction='ignore', lineterminator='\n')
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        with (out_dir / 'pnl' / f'{r["fold"]}.csv').open('w', newline='') as f:
            w = csv.writer(f, lineterminator='\n')
            w.writerow(['time_s', 'pnl'])
            w.writerows((f'{t:.3f}', f'{v:.2f}') for t, v in r['curve'])

    def fmt(v):
        return 'n/a' if v is None else f'{v:,.2f}' if isinstance(v, float) else f'{v:,}'
    print('\nWalk-forward summary')
    for k, v in summary.items():
        print(f'  {k:<24} {fmt(v)}')
    if not args.no_report:
        from walkforward.report import write_report
        print(f'report={write_report(out_dir)}')
    print(f'out_dir={out_dir}')


if __name__ == '__main__':
    main()
