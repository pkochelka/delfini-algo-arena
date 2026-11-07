"""Cut a backtest CSV into walk-forward folds, each a standalone tape the simulator can replay.

Eval windows tile [t0 + warmup, end) evenly; fold k's file holds [eval_start - warmup, eval_end) and
starts with a book snapshot: the market orders resting at the file's first timestamp (remaining
volume after any crossing), re-inserted at that timestamp in their original price-time priority, so
cancels and trades against older orders still work. Timestamps stay absolute.

The same layout as `strategies/peter/split_data.py --folds`, so fold files are interchangeable.
"""

from __future__ import annotations

import bisect
import csv
import hashlib
import json
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

HEADER = ['Time', 'Operation', 'OrderId', 'Side', 'Volume', 'Price', 'Lifespan']
T, OP, OID, SIDE, VOL, PX, LIFE = range(7)
CACHE_ROOT = Path('data/.cache/walkforward')


@dataclass(frozen=True)
class Fold:
    name: str
    path: str
    start_s: float  # first timestamp in the file (warmup begins)
    eval_start_s: float  # PnL counts from here
    end_s: float
    rows: int
    seeded: int  # resting orders re-inserted at start_s

    @property
    def eval_start_us(self) -> int:
        return int(round(self.eval_start_s * 1e6))


class _Book:
    """Minimal price-time book for market orders only, mirroring the simulator's matching."""

    def __init__(self) -> None:
        self.levels: dict[str, dict[int, deque[list]]] = {'B': {}, 'A': {}}
        self.orders: dict[str, list] = {}  # id -> [id, side, price, remaining, lifespan, seq]
        self.seq = 0

    def insert(self, oid: str, side: str, volume: int, price: int, lifespan: str) -> None:
        opp = 'A' if side == 'B' else 'B'
        book = self.levels[opp]
        while volume > 0 and book:
            best = min(book) if opp == 'A' else max(book)
            if (side == 'B' and best > price) or (side == 'A' and best < price):
                break
            queue = book[best]
            while volume > 0 and queue:
                resting = queue[0]
                traded = min(volume, resting[3])
                volume -= traded
                resting[3] -= traded
                if resting[3] == 0:
                    queue.popleft()
                    del self.orders[resting[0]]
            if not queue:
                del book[best]
        if volume > 0 and lifespan != 'F':
            self.seq += 1
            order = [oid, side, price, volume, lifespan, self.seq]
            self.orders[oid] = order
            self.levels[side].setdefault(price, deque()).append(order)

    def cancel(self, oid: str) -> None:
        order = self.orders.pop(oid, None)
        if order is None:
            return
        queue = self.levels[order[1]][order[2]]
        queue.remove(order)
        if not queue:
            del self.levels[order[1]][order[2]]

    def apply(self, row: list[str]) -> None:
        if row[OP] == 'Insert':
            self.insert(row[OID], row[SIDE], int(row[VOL]), int(row[PX]), row[LIFE])
        else:
            self.cancel(row[OID])

    def snapshot(self) -> list[list]:
        return [list(o) for o in sorted(self.orders.values(), key=lambda o: o[5])]


def _snapshots_at(rows: list[list[str]], times: list[float]) -> list[list[list]]:
    """Book snapshots just before each time in `times` (ascending), in one pass over `rows`."""
    book, out, i = _Book(), [], 0
    for t in times:
        while i < len(rows) and float(rows[i][T]) < t:
            book.apply(rows[i])
            i += 1
        out.append(book.snapshot())
    return out


def _cache_dir(backtest: Path, n_folds: int, warmup_s: float, end_frac: float) -> Path:
    st = backtest.stat()
    key = json.dumps([str(backtest.resolve()), st.st_size, st.st_mtime_ns, n_folds, warmup_s, end_frac])
    return CACHE_ROOT / f'{backtest.stem}-{hashlib.sha1(key.encode()).hexdigest()[:10]}'


def make_folds(backtest: Path, n_folds: int, warmup_s: float = 0.0, end_frac: float = 1.0) -> list[Fold]:
    """Write (or reuse cached) fold files for `backtest` and return their descriptions."""
    out_dir = _cache_dir(backtest, n_folds, warmup_s, end_frac)
    meta_path = out_dir / 'folds.json'
    if meta_path.exists():
        return [Fold(**f) for f in json.loads(meta_path.read_text())['folds']]

    with backtest.open(newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        if header != HEADER:
            raise ValueError(f'unexpected header {header}')
        rows = list(reader)
    times = [float(r[T]) for r in rows]
    t0, t1 = times[0], times[-1]
    end = t0 + (t1 - t0) * end_frac
    first = t0 + warmup_s
    if first >= end:
        raise ValueError(f'warmup {warmup_s}s leaves nothing to evaluate in {t1 - t0:.0f}s of data')
    step = (end - first) / n_folds
    evals = [(first + k * step, first + (k + 1) * step) for k in range(n_folds)]
    evals[-1] = (evals[-1][0], end if end_frac < 1 else t1 + 1e-6)  # keep the very last row
    starts = [max(t0, a - warmup_s) for a, _ in evals]

    out_dir.mkdir(parents=True, exist_ok=True)
    folds = []
    for k, ((a, b), s, snap) in enumerate(zip(evals, starts, _snapshots_at(rows, starts), strict=True), 1):
        path = out_dir / f'wf{k:02d}.csv'
        lo, hi = bisect.bisect_left(times, s), bisect.bisect_left(times, b)
        with path.open('w', newline='') as f:
            w = csv.writer(f, lineterminator='\n')
            w.writerow(HEADER)
            for oid, side, price, vol, life, _ in snap:
                w.writerow([f'{s:.6f}', 'Insert', oid, side, vol, price, life])
            w.writerows(rows[lo:hi])
        folds.append(Fold(path.stem, str(path), s, a, b, hi - lo, len(snap)))
    meta_path.write_text(json.dumps({'source': str(backtest), 'warmup_s': warmup_s, 'end_frac': end_frac,
                                     'folds': [asdict(f) for f in folds]}, indent=2))
    return folds
