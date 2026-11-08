"""Backtest CSV -> typed columns, cached as parquet next to the CSV in `.cache/` (data/ is git-ignored).

The columns reproduce `ExchangeSimulator.iter_backtest` exactly: `t_us = int(round(float(Time) * 1e6))`
(np.rint and round() both round half to even on the same float64), the same validation (unsupported
operations, bad sides, non-positive volume/price, negative time all raise), and the order ids as the
same strings. `verify()` checks that row by row against the reference parser.

    uv run python -m fastsim.tape data/round3.csv data/synth/*.csv [--verify]
"""

from __future__ import annotations

import argparse
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

TAPE_VERSION = 1
OP_INSERT, OP_CANCEL = 0, 1
SIDE_BUY, SIDE_SELL, SIDE_NONE = 0, 1, -1
_MEMORY_SLOTS = 3  # tapes kept in memory per process (~100 MB each for a 2.4M-row round)


@dataclass(frozen=True, slots=True)
class Tape:
    """A backtest as numpy columns, one entry per CSV row.

    `order_id` is int64 when every id is the canonical decimal string of an integer (then
    `str(order_id[i])` is the CSV string), otherwise an object array of the CSV strings.
    Cancel rows have side -1 and volume/price 0.
    """

    t_us: np.ndarray  # int64
    op: np.ndarray  # uint8, OP_INSERT / OP_CANCEL
    order_id: np.ndarray  # int64, or object (str)
    side: np.ndarray  # int8, SIDE_BUY / SIDE_SELL / SIDE_NONE
    volume: np.ndarray  # int64
    price: np.ndarray  # int64
    ioc: np.ndarray  # bool

    def __len__(self) -> int:
        return len(self.t_us)

    @property
    def numeric_ids(self) -> bool:
        return self.order_id.dtype != object


def cache_path(csv_path: Path) -> Path:
    """Where the columns of `csv_path` are cached; the key changes with the file's size, mtime and TAPE_VERSION."""
    stat = csv_path.stat()
    return csv_path.parent / '.cache' / f'{csv_path.stem}-{stat.st_size}-{stat.st_mtime_ns}-v{TAPE_VERSION}.parquet'


def parse_csv(csv_path: Path) -> Tape:
    """Read and validate the CSV, with the reference parser's semantics."""
    header = csv_path.open('r', newline='').readline().strip().split(',')
    types = {'Time': pa.float64(), 'Operation': pa.string(), 'OrderId': pa.string(), 'Side': pa.string(),
             'Volume': pa.int64(), 'Price': pa.int64(), 'Lifespan': pa.string()}
    table = pacsv.read_csv(
        csv_path,
        convert_options=pacsv.ConvertOptions(column_types={k: v for k, v in types.items() if k in header},
                                             strings_can_be_null=False),
    )
    n = table.num_rows
    t = table['Time'].to_numpy()
    t_us = np.rint(t * 1_000_000).astype(np.int64)

    ops = table['Operation'].to_numpy(zero_copy_only=False)
    is_ins = ops == 'Insert'
    is_can = ops == 'Cancel'
    bad = ~(is_ins | is_can)
    if bad.any():
        raise ValueError(f'Unsupported operation: {ops[np.argmax(bad)]}')
    op = np.where(is_ins, OP_INSERT, OP_CANCEL).astype(np.uint8)

    sides = table['Side'].to_numpy(zero_copy_only=False)
    side = np.full(n, SIDE_NONE, np.int8)
    side[is_ins & (sides == 'B')] = SIDE_BUY
    side[is_ins & (sides == 'A')] = SIDE_SELL
    if (is_ins & (side == SIDE_NONE)).any():
        raise ValueError(f'Invalid side in insert row {int(np.argmax(is_ins & (side == SIDE_NONE)))}')

    def ints(name: str) -> np.ndarray:
        col = table[name]
        if col.null_count and pc.any(pc.and_(pc.is_null(col), pa.array(is_ins))).as_py():
            raise ValueError(f'Missing {name} in an insert row')
        return col.fill_null(0).to_numpy().astype(np.int64)

    volume, price = ints('Volume'), ints('Price')
    volume[~is_ins] = 0
    price[~is_ins] = 0
    if (t_us < 0).any() or (is_ins & ((volume <= 0) | (price <= 0))).any():
        raise ValueError('Invalid MarketEvent fields (negative time or non-positive volume/price)')

    if 'Lifespan' in header:
        ioc = is_ins & (table['Lifespan'].to_numpy(zero_copy_only=False) == 'F')
    else:
        ioc = np.zeros(n, bool)

    id_strings = table['OrderId'].to_numpy(zero_copy_only=False)
    order_id: np.ndarray = id_strings.astype(object)
    try:
        as_int = np.array([int(s) for s in id_strings], dtype=np.int64)
        if all(str(v) == s for v, s in zip(as_int.tolist(), id_strings.tolist())):
            order_id = as_int
    except (ValueError, OverflowError):
        pass
    return Tape(t_us=t_us, op=op, order_id=order_id, side=side, volume=volume, price=price, ioc=ioc)


def _write(tape: Tape, target: Path) -> None:
    oid = pa.array(tape.order_id) if tape.numeric_ids else pa.array(tape.order_id.tolist(), pa.string())
    table = pa.table({'t_us': tape.t_us, 'op': tape.op, 'order_id': oid, 'side': tape.side,
                      'volume': tape.volume, 'price': tape.price, 'ioc': tape.ioc})
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f'{target.name}.{time.monotonic_ns()}.tmp')  # concurrent builders don't clash
    pq.write_table(table, tmp, compression='zstd')
    tmp.replace(target)


def _read(path: Path) -> Tape:
    table = pq.read_table(path)
    col = {name: table[name].to_numpy(zero_copy_only=False) for name in table.column_names}
    oid = col['order_id']
    if oid.dtype != np.int64:
        oid = oid.astype(object)
    return Tape(t_us=col['t_us'], op=col['op'], order_id=oid, side=col['side'],
                volume=col['volume'], price=col['price'], ioc=col['ioc'])


def ensure_cache(csv_path: str | Path, *, verbose: bool = False) -> Path:
    """Build the parquet cache of `csv_path` if it is missing or stale; return its path."""
    csv_path = Path(csv_path).resolve()
    target = cache_path(csv_path)
    if target.exists():
        return target
    started = time.perf_counter()
    tape = parse_csv(csv_path)
    for stale in target.parent.glob(f'{csv_path.stem}-*.parquet'):
        stale.unlink(missing_ok=True)
    _write(tape, target)
    if verbose:
        print(f'{csv_path.name}: {len(tape):,} rows -> {target.name} in {time.perf_counter() - started:.1f}s', flush=True)
    return target


_memory: OrderedDict[Path, Tape] = OrderedDict()


def load_tape(csv_path: str | Path) -> Tape:
    """Return the columns of `csv_path`: from this process's memory, else the parquet cache, else the CSV."""
    target = ensure_cache(csv_path)
    tape = _memory.get(target)
    if tape is None:
        tape = _memory[target] = _read(target)
        while len(_memory) > _MEMORY_SLOTS:
            _memory.popitem(last=False)
    else:
        _memory.move_to_end(target)
    return tape


def verify(csv_path: str | Path) -> int:
    """Compare every row of the cached columns with the reference parser; return the row count or raise."""
    from simulator.models import MarketOperation, OrderType, Side
    from simulator.simulator import ExchangeSimulator

    tape = load_tape(csv_path)
    side_of = {SIDE_BUY: Side.BUY, SIDE_SELL: Side.SELL}
    t_us, op, side, vol, px, ioc = (a.tolist() for a in (tape.t_us, tape.op, tape.side, tape.volume, tape.price, tape.ioc))
    oid = tape.order_id.tolist()
    n = 0
    for i, ev in enumerate(ExchangeSimulator.iter_backtest(Path(csv_path))):
        n += 1
        ok = ev.timestamp_us == t_us[i] and ev.order_id == str(oid[i])
        if ev.operation is MarketOperation.INSERT:
            ok = ok and op[i] == OP_INSERT and ev.side is side_of[side[i]] and ev.volume == vol[i] \
                and ev.price == px[i] and (ev.order_type is OrderType.IOC) == ioc[i]
        else:
            ok = ok and op[i] == OP_CANCEL
        if not ok:
            raise AssertionError(f'row {i} differs: reference {ev!r}, cached '
                                 f'{(t_us[i], op[i], oid[i], side[i], vol[i], px[i], ioc[i])}')
    if n != len(tape):
        raise AssertionError(f'row count differs: reference {n}, cached {len(tape)}')
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description='Build (and optionally verify) the parquet caches of backtest CSVs')
    ap.add_argument('csv', nargs='+', type=Path)
    ap.add_argument('--verify', action='store_true', help='compare every row with the reference CSV parser')
    args = ap.parse_args()
    for path in args.csv:
        ensure_cache(path, verbose=True)
        if args.verify:
            started = time.perf_counter()
            print(f'{path.name}: {verify(path):,} rows identical ({time.perf_counter() - started:.1f}s)', flush=True)


if __name__ == '__main__':
    main()
