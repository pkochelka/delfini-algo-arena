"""Disk cache of replayed tapes, next to the CSV in `.cache/` (data/ is git-ignored)."""

from __future__ import annotations

import pickle
import time
from pathlib import Path

from analysis.replay import REPLAY_VERSION, Tape, replay_csv


def cache_path(csv_path: Path) -> Path:
    """Where the replay of `csv_path` is cached; the key changes with the file's size, mtime and REPLAY_VERSION."""
    stat = csv_path.stat()
    return csv_path.parent / '.cache' / f'{csv_path.stem}-{stat.st_size}-{stat.st_mtime_ns}-v{REPLAY_VERSION}.pkl'


def load_tape(csv_path: str | Path, *, refresh: bool = False, verbose: bool = True) -> Tape:
    """Return the replayed tape for `csv_path`, from the cache when it is there and current."""
    csv_path = Path(csv_path)
    target = cache_path(csv_path)
    if target.exists() and not refresh:
        with target.open('rb') as file:
            return pickle.load(file)

    started = time.perf_counter()
    if verbose:
        print(f'Replaying {csv_path} ...', flush=True)
    tape = replay_csv(csv_path)
    if verbose:
        print(
            f'  {len(tape.events):,} events, {len(tape.trades):,} trades in {time.perf_counter() - started:.1f}s',
            flush=True,
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    for stale in target.parent.glob(f'{csv_path.stem}-*.pkl'):
        stale.unlink()
    tmp = target.with_suffix('.tmp')
    with tmp.open('wb') as file:
        pickle.dump(tape, file, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(target)
    return tape
