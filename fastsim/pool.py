"""Run many backtests in worker processes, in-process in each worker (no `uv run` subprocess per job).

    from fastsim.pool import Job, run_many
    outs = run_many([Job('strategies/x.py', 'data/round3.csv', {'eta': 0.2}), ...])
    # each out: {'job': Job, 'result': {...result.json...}, 'curve': [(elapsed_s, pnl), ...], 'secs': float}
    #       or: {'job': Job, 'err': traceback}

Tape caches are built in the parent first, so workers never race to convert the same CSV. Each worker
keeps a few tapes in memory between jobs. `params` updates the strategy module's PARAMS after it has
run (what appending `PARAMS.update(...)` to a temp copy of the source used to do), so the strategy file
path stays stable, which numba's on-disk cache needs.
"""

from __future__ import annotations

import concurrent.futures as cf
import os
import time
import traceback
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastsim.tape import ensure_cache

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Job:
    strategy: str  # path, relative to the current directory or absolute
    tape: str  # path, relative to the repo root or absolute
    params: dict[str, Any] = field(default_factory=dict)
    jit: bool = False


def default_workers() -> int:
    return max(1, (os.cpu_count() or 2) - 1)


def _tape_path(tape: str) -> Path:
    path = Path(tape)
    return path if path.is_absolute() else ROOT / path


def run_job(job: Job) -> dict[str, Any]:
    """Run one job in this process; never raises (errors come back as 'err')."""
    from fastsim.engine import run_backtest

    started = time.perf_counter()
    try:
        result, curve = run_backtest(Path(job.strategy).resolve(), _tape_path(job.tape), job.params or None,
                                     jit=job.jit)
    except Exception:
        return {'job': job, 'err': traceback.format_exc()[-4000:], 'secs': time.perf_counter() - started}
    return {'job': job, 'result': result.model_dump(mode='json'), 'curve': curve,
            'secs': time.perf_counter() - started}


def run_many(
    jobs: Iterable[Job],
    workers: int | None = None,
    on_result: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Run `jobs` on a process pool; return the outputs in job order (`on_result` sees each as it finishes)."""
    jobs = list(jobs)
    if not jobs:
        return []
    workers = min(workers or default_workers(), len(jobs))
    for tape in sorted({_tape_path(j.tape) for j in jobs}):
        ensure_cache(tape, verbose=True)
    if workers == 1:
        outs = []
        for job in jobs:
            outs.append(run_job(job))
            if on_result:
                on_result(outs[-1])
        return outs
    outs: list[dict[str, Any] | None] = [None] * len(jobs)
    with cf.ProcessPoolExecutor(workers) as ex:
        futures = {ex.submit(run_job, job): i for i, job in enumerate(jobs)}
        for fut in cf.as_completed(futures):
            out = outs[futures[fut]] = fut.result()
            if on_result:
                on_result(out)
    return outs  # type: ignore[return-value]
