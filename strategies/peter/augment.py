"""Write augmented copies of backtest CSVs: transformed, noised, and with injected events.

Usage:
    uv run python strategies/peter/augment.py data/splits/train.csv data/splits/dev.csv
    uv run python strategies/peter/augment.py data/splits/dev.csv --levels 0.5 1 2 4 --seed 7

Each input gets its own augmented files (data/splits/aug/<name>_<tag>.csv), so augmentations of
train never leak into dev. Every output is a valid backtest; a cancel for an order that is gone
(e.g. eaten by an injected taker) is ignored by the simulator.

Deterministic transforms
    mirror      swap B/A and reflect prices p -> lo + hi - p (an uptrend becomes a downtrend)
    pscale k    p -> p0 + k*(p - p0), integer k (spreads and moves k times wider in ticks)
    tscale s    t -> t0 + s*(t - t0) (fewer or more events inside the 5 ms latency)
    vscale s    v -> max(1, round(s*v)) (thinner or thicker book)
    iocscale s  the same for aggressive (F) orders only (bigger sweeps, same book)
    cscale s    every cancel moves to insert + s*(cancel - insert) (slower / faster queue turnover)
    drift r     inserted prices shifted by round(r * (t - t0) / 1000): a trend of r ticks per 1000 s
                (orders resting from before a shift can now be crossed, as in a real trending market)

Noise, strength set by the argument (seeded)
    vnoise σ    volumes times a mean-one lognormal(σ)
    tnoise σ    every gap between events times a mean-one lognormal(σ), total duration kept
    pnoise p    a resting insert moves one tick with probability p (outward if inward would cross)
    drop p      a resting insert and its cancel are removed with probability p

Injected events, rate per second (Poisson, priced off the book replayed at that moment)
    takers r    IOC at the touch, random side, size drawn from the tape's IOC sizes
    makers r    resting order 1-3 ticks behind the touch, cancelled after Exp(2 s)
    flicker r   5x-size order joining the touch, cancelled after 20-100 ms (fake imbalance)
    shocks r    IOC sweeping the top 3 levels of one side (a liquidity shock)

Presets: the deterministic tags, plus for every noise level L in --levels:
    noise{L}    vnoise 0.3L, tnoise 0.3L, pnoise 0.1L, drop 0.05L
    takers{L}   takers 0.1L      makers{L}  makers 0.5L
    flicker{L}  flicker 0.05L    shocks{L}  shocks 0.005L
    all{L}      everything above at level L
"""

from __future__ import annotations

import argparse
import csv
import heapq
import math
import random
from pathlib import Path

from split_data import Book

HEADER = ['Time', 'Operation', 'OrderId', 'Side', 'Volume', 'Price', 'Lifespan']

DETERMINISTIC: dict[str, list[tuple[str, float]]] = {
    'mirror': [('mirror', 0)],
    'pscale2': [('pscale', 2)],
    'tscale0.25': [('tscale', 0.25)],
    'tscale4': [('tscale', 4)],
    'vscale0.3': [('vscale', 0.3)],
    'vscale3': [('vscale', 3)],
    'mirror_pscale2': [('mirror', 0), ('pscale', 2)],
    'drift10': [('drift', 10)],
    'drift30': [('drift', 30)],
    'drift100': [('drift', 100)],
    'drift-30': [('drift', -30)],
    # intermediate book depths and combined shifts (queue_mm robustness, see queue_hypotheses.md)
    'vscale0.6': [('vscale', 0.6)],
    'vscale1.7': [('vscale', 1.7)],
    'vscale2': [('vscale', 2)],
    'pscale2_vscale0.3': [('pscale', 2), ('vscale', 0.3)],
    'pscale2_vscale3': [('pscale', 2), ('vscale', 3)],
    'vscale0.3_takers3': [('vscale', 0.3), ('takers', 0.3)],
    'tscale4_vscale3_drift30': [('tscale', 4), ('vscale', 3), ('drift', 30)],
    'iocscale0.5': [('iocscale', 0.5)],
    'iocscale2': [('iocscale', 2)],
    'iocscale3': [('iocscale', 3)],
    'cscale0.5': [('cscale', 0.5)],
    'cscale2': [('cscale', 2)],
}


def noise_presets(level: float) -> dict[str, list[tuple[str, float]]]:
    noise = [('vnoise', 0.3 * level), ('tnoise', 0.3 * level), ('pnoise', 0.1 * level), ('drop', 0.05 * level)]
    events = {
        'takers': [('takers', 0.1 * level)],
        'makers': [('makers', 0.5 * level)],
        'flicker': [('flicker', 0.05 * level)],
        'shocks': [('shocks', 0.005 * level)],
    }
    tag = f'{level:g}'
    out = {f'noise{tag}': noise}
    out.update({f'{k}{tag}': v for k, v in events.items()})
    out[f'all{tag}'] = noise + [s for v in events.values() for s in v]
    return out


BOOK_STEPS = {'pnoise', 'takers', 'makers', 'flicker', 'shocks'}


# ----------------------------------------------------------------- simple steps
def simple_step(rows: list[dict], name: str, arg: float, rng: random.Random) -> list[dict]:
    prices = [int(r['Price']) for r in rows if r['Price']]
    lo, hi, p0 = min(prices), max(prices), prices[0]
    t0 = float(rows[0]['Time'])
    if name == 'drop':
        dropped = {r['OrderId'] for r in rows
                   if r['Operation'] == 'Insert' and r['Lifespan'] == 'G' and rng.random() < arg}
        return [r for r in rows if r['OrderId'] not in dropped]
    if name == 'iocscale':  # aggressive (F) volumes only: bigger/smaller sweeps, same book
        for r in rows:
            if r['Operation'] == 'Insert' and r['Lifespan'] == 'F':
                r['Volume'] = str(max(1, round(arg * int(r['Volume']))))
        return rows
    if name == 'cscale':  # resting orders live arg times as long before their cancel (queue turnover)
        born = {r['OrderId']: float(r['Time']) for r in rows if r['Operation'] == 'Insert'}
        for r in rows:
            if r['Operation'] == 'Cancel' and r['OrderId'] in born:
                tb = born[r['OrderId']]
                r['Time'] = f"{tb + arg * (float(r['Time']) - tb):.6f}"
        rows.sort(key=lambda r: float(r['Time']))  # stable: same-time rows keep their order
        return rows

        times = [float(r['Time']) for r in rows]
        gaps = [b - a for a, b in zip(times, times[1:])]
        noisy = [g * math.exp(rng.gauss(0, arg) - arg * arg / 2) for g in gaps]
        scale = sum(gaps) / sum(noisy) if sum(noisy) > 0 else 1.0
        t = times[0]
        for r, g in zip(rows[1:], noisy):
            t += g * scale
            r['Time'] = f'{t:.6f}'
        return rows
    for r in rows:
        if name == 'mirror':
            if r['Price']:
                r['Price'] = str(lo + hi - int(r['Price']))
            if r['Side']:
                r['Side'] = 'A' if r['Side'] == 'B' else 'B'
        elif name == 'pscale':
            k = int(arg)
            if k < 1 or k != arg:
                raise ValueError('pscale needs a positive integer')
            if r['Price']:
                r['Price'] = str(p0 + k * (int(r['Price']) - p0))
        elif name == 'tscale':
            r['Time'] = f"{t0 + arg * (float(r['Time']) - t0):.6f}"
        elif name == 'drift':
            if r['Price']:
                r['Price'] = str(int(r['Price']) + round(arg * (float(r['Time']) - t0) / 1000))
        elif name == 'vscale':
            if r['Volume']:
                r['Volume'] = str(max(1, round(arg * int(r['Volume']))))
        elif name == 'vnoise':
            if r['Volume']:
                f = math.exp(rng.gauss(0, arg) - arg * arg / 2)
                r['Volume'] = str(max(1, round(f * int(r['Volume']))))
        else:
            raise ValueError(f'unknown transform {name}')
    if name == 'pscale' and min(int(r['Price']) for r in rows if r['Price']) <= 0:
        raise ValueError('pscale produced a non-positive price')
    return rows


# ------------------------------------------------------ book-dependent steps
def best(book: Book, side: str) -> int | None:
    lv = book.levels[side]
    if not lv:
        return None
    return max(lv) if side == 'B' else min(lv)


def poisson_times(rng: random.Random, rate: float, t0: float, t1: float) -> list[float]:
    out, t = [], t0
    if rate <= 0:
        return out
    while True:
        t += rng.expovariate(rate)
        if t >= t1:
            return out
        out.append(t)


def book_steps(rows: list[dict], cfg: dict[str, float], rng: random.Random) -> list[dict]:
    """Replay the tape through a book, jittering resting inserts and injecting new events."""
    t0, t1 = float(rows[0]['Time']), float(rows[-1]['Time'])
    ioc_vols = [int(r['Volume']) for r in rows if r['Operation'] == 'Insert' and r['Lifespan'] == 'F'] or [10]
    g_vols = [int(r['Volume']) for r in rows if r['Operation'] == 'Insert' and r['Lifespan'] == 'G'] or [10]
    next_id = max(int(r['OrderId']) for r in rows if r['OrderId'].isdigit()) + 1

    heap: list[tuple[float, int, str, dict]] = []
    seq = 0
    for kind in ('takers', 'makers', 'flicker', 'shocks'):
        for t in poisson_times(rng, cfg.get(kind, 0.0), t0, t1):
            heap.append((t, seq, kind, {}))
            seq += 1
    heapq.heapify(heap)

    book = Book()
    out: list[dict] = []

    def emit(t: float, op: str, oid: str, side: str = '', vol: int | None = None,
             price: int | None = None, life: str = '') -> None:
        out.append({'Time': f'{t:.6f}', 'Operation': op, 'OrderId': oid, 'Side': side,
                    'Volume': '' if vol is None else str(vol), 'Price': '' if price is None else str(price),
                    'Lifespan': life})
        if op == 'Insert':
            book.insert(oid, side, vol, price, life)
        else:
            book.cancel(oid)

    def inject(t: float, kind: str, payload: dict) -> None:
        nonlocal next_id, seq
        if kind == 'cancel':
            emit(t, 'Cancel', payload['oid'])
            return
        side = rng.choice('BA')
        opp = 'A' if side == 'B' else 'B'
        own_best, opp_best = best(book, side), best(book, opp)
        oid = str(next_id)
        next_id += 1
        out_dir = -1 if side == 'B' else 1  # direction away from the touch
        if kind == 'takers':
            if opp_best is None:
                return
            emit(t, 'Insert', oid, side, rng.choice(ioc_vols), opp_best, 'F')
        elif kind == 'shocks':
            if opp_best is None:
                return
            limit = opp_best - 2 * out_dir  # three levels into the opposite side
            in_range = (lambda p: p <= limit) if side == 'B' else (lambda p: p >= limit)
            vol = sum(o[3] for p, q in book.levels[opp].items() if in_range(p) for o in q)
            if vol > 0:
                emit(t, 'Insert', oid, side, vol, limit, 'F')
        elif kind in ('makers', 'flicker'):
            ref = own_best if own_best is not None else (opp_best + out_dir if opp_best is not None else None)
            if ref is None:
                return
            if kind == 'makers':
                price, vol, life = ref + out_dir * rng.randint(1, 3), rng.choice(g_vols), rng.expovariate(0.5)
            else:
                price, vol, life = ref, 5 * rng.choice(g_vols), rng.uniform(0.02, 0.1)
            if price <= 0:
                return
            emit(t, 'Insert', oid, side, vol, price, 'G')
            heapq.heappush(heap, (t + life, seq, 'cancel', {'oid': oid}))
            seq += 1

    p_noise = cfg.get('pnoise', 0.0)
    for r in rows:
        t = float(r['Time'])
        while heap and heap[0][0] < t:
            ti, _, kind, payload = heapq.heappop(heap)
            inject(ti, kind, payload)
        if r['Operation'] == 'Insert':
            price = int(r['Price'])
            if r['Lifespan'] == 'G' and p_noise and rng.random() < p_noise:
                opp_best = best(book, 'A' if r['Side'] == 'B' else 'B')
                shift = rng.choice((-1, 1))
                crosses = opp_best is not None and (
                    price + shift >= opp_best if r['Side'] == 'B' else price + shift <= opp_best)
                if crosses:
                    shift = -1 if r['Side'] == 'B' else 1
                price = max(1, price + shift)
            emit(t, 'Insert', r['OrderId'], r['Side'], int(r['Volume']), price, r['Lifespan'])
        else:
            emit(t, 'Cancel', r['OrderId'])
    # whatever is left is cancels due after the tape ends: those orders just rest to the end
    return out


def transform(rows: list[dict], steps: list[tuple[str, float]], rng: random.Random) -> list[dict]:
    rows = [dict(r) for r in rows]
    simple = [(n, a) for n, a in steps if n not in BOOK_STEPS]
    booked = {n: a for n, a in steps if n in BOOK_STEPS}
    for name, arg in simple:
        rows = simple_step(rows, name, arg, rng)
    if booked:
        rows = book_steps(rows, booked, rng)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('inputs', type=Path, nargs='+')
    parser.add_argument('--out-dir', type=Path, default=Path('data/splits/aug'))
    parser.add_argument('--levels', type=float, nargs='+', default=[1, 3], help='noise levels for the noise presets')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--only', nargs='+', help='subset of augment tags to write')
    args = parser.parse_args()
    augments = dict(DETERMINISTIC)
    for level in args.levels:
        augments.update(noise_presets(level))
    if args.only:
        augments = {k: v for k, v in augments.items() if k in args.only}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for path in args.inputs:
        with path.open(newline='') as f:
            rows = list(csv.DictReader(f))
        for tag, steps in augments.items():
            rng = random.Random(f'{args.seed}-{path.stem}-{tag}')
            out_rows = transform(rows, steps, rng)
            out = args.out_dir / f'{path.stem}_{tag}.csv'
            with out.open('w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=HEADER, lineterminator='\n')
                w.writeheader()
                w.writerows(out_rows)
            print(f'{path.stem} {tag:16} rows={len(out_rows):>6} (was {len(rows)}) -> {out}')


if __name__ == '__main__':
    main()
