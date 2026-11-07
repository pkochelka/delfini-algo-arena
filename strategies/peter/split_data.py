"""Split a backtest CSV into chronological train/dev/test files.

Usage:
    uv run python strategies/peter/split_data.py --backtest data/sample.csv --ratios 2 1 1

Each split covers a contiguous time window. Because orders inserted before a cut can be
cancelled or traded against after it, every split after the first starts with a book
snapshot: the market orders resting at the cut (with their remaining volume after any
crossing), re-inserted at the cut timestamp in original price-time priority. Cancels for
orders that no longer exist in a split are dropped. Timestamps stay absolute.
"""

from __future__ import annotations

import argparse
import csv
from collections import deque
from pathlib import Path

HEADER = ['Time', 'Operation', 'OrderId', 'Side', 'Volume', 'Price', 'Lifespan']


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

    def cancel(self, oid: str) -> bool:
        order = self.orders.pop(oid, None)
        if order is None:
            return False
        queue = self.levels[order[1]][order[2]]
        queue.remove(order)
        if not queue:
            del self.levels[order[1]][order[2]]
        return True

    def snapshot(self) -> list[list]:
        return sorted(self.orders.values(), key=lambda o: o[5])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--backtest', type=Path, default=Path('data/sample.csv'))
    parser.add_argument('--ratios', type=float, nargs='+', default=[2, 1, 1])
    parser.add_argument('--names', nargs='+', default=['train', 'dev', 'test'])
    parser.add_argument('--out-dir', type=Path, default=Path('data/splits'))
    args = parser.parse_args()
    if len(args.ratios) != len(args.names):
        parser.error('--ratios and --names must have the same length')

    with args.backtest.open(newline='') as f:
        rows = list(csv.DictReader(f))
    t0, t1 = float(rows[0]['Time']), float(rows[-1]['Time'])
    total = sum(args.ratios)
    cuts, acc = [], 0.0
    for r in args.ratios[:-1]:
        acc += r
        cuts.append(t0 + (t1 - t0) * acc / total)
    cuts.append(float('inf'))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    book = Book()
    i = 0
    start = t0
    for name, cut in zip(args.names, cuts):
        path = args.out_dir / f'{name}.csv'
        n_seed = n_rows = n_dropped = 0
        with path.open('w', newline='') as f:
            w = csv.writer(f, lineterminator='\n')
            w.writerow(HEADER)
            for oid, side, price, vol, life, _ in book.snapshot():
                w.writerow([f'{start:.6f}', 'Insert', oid, side, vol, price, life])
                n_seed += 1
            alive = set(book.orders)
            while i < len(rows) and float(rows[i]['Time']) < cut:
                row = rows[i]
                i += 1
                if row['Operation'] == 'Insert':
                    book.insert(row['OrderId'], row['Side'], int(row['Volume']), int(row['Price']), row['Lifespan'])
                    alive.add(row['OrderId'])
                else:
                    book.cancel(row['OrderId'])
                    if row['OrderId'] not in alive:
                        n_dropped += 1
                        continue
                w.writerow([row[h] for h in HEADER])
                n_rows += 1
        end = min(cut, t1)
        print(f'{name}: {start:.1f}s -> {end:.1f}s  rows={n_rows}  seeded={n_seed}  dropped_cancels={n_dropped}  -> {path}')
        start = cut


if __name__ == '__main__':
    main()
