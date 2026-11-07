"""Split a backtest CSV chronologically: train/dev/test, or walk-forward folds with a warmup prefix.

Usage:
    # contiguous splits (default 2:1:1)
    uv run python strategies/peter/split_data.py --backtest data/sample_v2.csv --out-dir data/splits_v2
    uv run python strategies/peter/split_data.py --ratios 1 1 1

    # walk-forward folds over the first 75% (the last 25% = test stays untouched)
    uv run python strategies/peter/split_data.py --backtest data/sample_v2.csv \\
        --folds 12 --warmup-s 60 --end-frac 0.75 --out-dir data/splits_v2/wf

Every output file starts with a book snapshot: the market orders resting at the file's first
timestamp (remaining volume after any crossing), re-inserted at that timestamp in their original
price-time priority, so cancels and trades against older orders still work. Timestamps stay absolute.

Fold mode: eval windows tile [t0 + warmup, end) evenly; fold k's file holds [eval_start - warmup,
eval_end). The strategy should learn on the prefix but only trade from eval_start; folds.json records
that as `trade_start_us` (arb_mm's PARAMS key of the same name). Overlapping prefixes are intended.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
from collections import deque
from pathlib import Path

HEADER = ['Time', 'Operation', 'OrderId', 'Side', 'Volume', 'Price', 'Lifespan']
T, OP, OID, SIDE, VOL, PX, LIFE = range(7)


class Book:
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
        """Resting orders in original arrival order (keeps queue priority within each level)."""
        return [list(o) for o in sorted(self.orders.values(), key=lambda o: o[5])]


def snapshots_at(rows: list[list[str]], times: list[float]) -> list[list[list]]:
    """Book snapshots just before each time in `times` (ascending), in one pass over `rows`."""
    book, out, i = Book(), [], 0
    for t in times:
        while i < len(rows) and float(rows[i][T]) < t:
            book.apply(rows[i])
            i += 1
        out.append(book.snapshot())
    return out


def write_window(path: Path, rows: list[list[str]], times: list[float], start: float, end: float,
                 snapshot: list[list]) -> int:
    lo, hi = bisect.bisect_left(times, start), bisect.bisect_left(times, end)
    with path.open('w', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(HEADER)
        for oid, side, price, vol, life, _ in snapshot:
            w.writerow([f'{start:.6f}', 'Insert', oid, side, vol, price, life])
        w.writerows(rows[lo:hi])
    return hi - lo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--backtest', type=Path, default=Path('data/sample.csv'))
    parser.add_argument('--out-dir', type=Path, default=Path('data/splits'))
    parser.add_argument('--ratios', type=float, nargs='+', default=[2, 1, 1])
    parser.add_argument('--names', nargs='+', default=['train', 'dev', 'test'])
    parser.add_argument('--folds', type=int, help='write N walk-forward folds instead of named splits')
    parser.add_argument('--warmup-s', type=float, default=60.0, help='fold mode: learn-only prefix per fold')
    parser.add_argument('--end-frac', type=float, default=0.75,
                        help='fold mode: folds cover [0, end_frac) of the duration; the rest is held out')
    args = parser.parse_args()

    with args.backtest.open(newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        if header != HEADER:
            raise ValueError(f'unexpected header {header}')
        rows = list(reader)
    times = [float(r[T]) for r in rows]
    t0, t1 = times[0], times[-1]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.folds:
        end = t0 + (t1 - t0) * args.end_frac
        first = t0 + args.warmup_s
        step = (end - first) / args.folds
        evals = [(first + k * step, first + (k + 1) * step) for k in range(args.folds)]
        starts = [max(t0, a - args.warmup_s) for a, _ in evals]
        snaps = snapshots_at(rows, starts)
        meta = []
        for k, ((a, b), s, snap) in enumerate(zip(evals, starts, snaps), 1):
            path = args.out_dir / f'wf{k:02d}.csv'
            n = write_window(path, rows, times, s, b, snap)
            meta.append({'name': path.stem, 'file': path.name, 'start_s': s, 'eval_start_s': a, 'end_s': b,
                         'trade_start_us': int(round(a * 1e6)), 'rows': n, 'seeded': len(snap)})
            print(f'{path.stem}: learn {s:.1f}s, trade {a:.1f}s -> {b:.1f}s  rows={n}  seeded={len(snap)}')
        meta_path = args.out_dir / 'folds.json'
        meta_path.write_text(json.dumps({'source': str(args.backtest), 'warmup_s': args.warmup_s,
                                         'end_frac': args.end_frac, 'held_out_from_s': end,
                                         'folds': meta}, indent=2))
        print(f'held out (untouched): {end:.1f}s -> {t1:.1f}s   meta -> {meta_path}')
        return

    if len(args.ratios) != len(args.names):
        parser.error('--ratios and --names must have the same length')
    total, acc, bounds = sum(args.ratios), 0.0, [t0]
    for r in args.ratios[:-1]:
        acc += r
        bounds.append(t0 + (t1 - t0) * acc / total)
    bounds.append(float('inf'))
    snaps = snapshots_at(rows, bounds[:-1])
    for name, s, e, snap in zip(args.names, bounds, bounds[1:], snaps):
        path = args.out_dir / f'{name}.csv'
        n = write_window(path, rows, times, s, e, snap)
        print(f'{name}: {s:.1f}s -> {min(e, t1):.1f}s  rows={n}  seeded={len(snap)}  -> {path}')


if __name__ == '__main__':
    main()
