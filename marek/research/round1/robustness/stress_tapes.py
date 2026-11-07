"""Stress tapes for robustness tests: strategies/peter/augment.py presets plus scenarios it lacks.

Usage:
    uv run python research/round1/robustness/stress_tapes.py data/splits_round1/wf/wf03.csv --out-dir data/stress
    uv run python research/round1/robustness/stress_tapes.py data/splits_round1/wf/wf03.csv --only toxic1 burst

Writes data/stress/<input stem>_<tag>.csv. Tags = augment.py's DETERMINISTIC + noise presets at
--levels, plus (all seeded):
    toxicL      informed IOC takers at 0.1*L /s: side = sign of the tape mid move over the next 2 s
                (skip if flat); size from the tape's IOC sizes. Uninformed `takersL` is the control.
    bigtakerK   every market IOC volume times K (larger aggressive orders, sweeps deeper)
    burst       three trend bursts: inserted prices shifted by a ramp of +-40 ticks over 60 s,
                kept afterwards (level shifts / news jumps); the shift sign alternates
    trendup / trenddown  a sustained one-way trend of 300 ticks over the window (drift with the
                augment.py semantics, ~ 250 ticks/1000 s on a 1200 s fold)
    volreg      second half of the window has price moves x2 (pscale 2 about the half-time mid,
                applied only after the midpoint: a volatility regime switch)
"""

from __future__ import annotations

import argparse
import csv
import heapq
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'strategies' / 'peter'))
import augment as A  # noqa: E402
from split_data import Book  # noqa: E402

from analysis.replay import replay_csv  # noqa: E402


def future_mid(path: Path):
    tape = replay_csv(path)
    q = tape.quotes
    q = q[q['mid'].notna()]
    t = (q['t_us'].to_numpy() / 1e6)
    m = q['mid'].to_numpy()
    import numpy as np

    def at(x):
        i = np.searchsorted(t, x, side='right') - 1
        return m[max(i, 0)]
    return at


def toxic_steps(rows, rate, rng, mid_at, horizon=2.0):
    """Like augment.book_steps with only informed takers injected."""
    t0, t1 = float(rows[0]['Time']), float(rows[-1]['Time'])
    ioc_vols = [int(r['Volume']) for r in rows if r['Operation'] == 'Insert' and r['Lifespan'] == 'F'] or [10]
    next_id = max(int(r['OrderId']) for r in rows if r['OrderId'].isdigit()) + 1
    heap = [(t, i) for i, t in enumerate(A.poisson_times(rng, rate, t0, t1))]
    heapq.heapify(heap)
    book = Book()
    out = []

    def emit(t, op, oid, side='', vol=None, price=None, life=''):
        out.append({'Time': f'{t:.6f}', 'Operation': op, 'OrderId': oid, 'Side': side,
                    'Volume': '' if vol is None else str(vol), 'Price': '' if price is None else str(price),
                    'Lifespan': life})
        if op == 'Insert':
            book.insert(oid, side, vol, price, life)
        else:
            book.cancel(oid)

    for r in rows:
        t = float(r['Time'])
        while heap and heap[0][0] < t:
            ti, _ = heapq.heappop(heap)
            d = mid_at(ti + horizon) - mid_at(ti)
            if d == 0:
                continue
            side = 'B' if d > 0 else 'A'
            opp = 'A' if side == 'B' else 'B'
            ob = A.best(book, opp)
            if ob is None:
                continue
            emit(ti, 'Insert', str(next_id), side, rng.choice(ioc_vols), ob, 'F')
            next_id += 1
        if r['Operation'] == 'Insert':
            emit(t, 'Insert', r['OrderId'], r['Side'], int(r['Volume']), int(r['Price']), r['Lifespan'])
        else:
            emit(t, 'Cancel', r['OrderId'])
    return out


def price_shift(rows, shift_fn):
    for r in rows:
        if r['Price']:
            r['Price'] = str(int(r['Price']) + int(round(shift_fn(float(r['Time'])))))
    return rows


def burst_fn(t0, t1, size=40, ramp=60.0):
    span = t1 - t0
    starts = [t0 + span * f for f in (0.2, 0.5, 0.8)]
    signs = [1, -1, 1]

    def f(t):
        s = 0.0
        for st, sg in zip(starts, signs):
            if t >= st:
                s += sg * size * min(1.0, (t - st) / ramp)
        return s
    return f


def volreg(rows):
    t0, t1 = float(rows[0]['Time']), float(rows[-1]['Time'])
    tm = (t0 + t1) / 2
    # reference price: last inserted price before the midpoint
    ref = None
    for r in rows:
        if float(r['Time']) >= tm:
            break
        if r['Price']:
            ref = int(r['Price'])
    for r in rows:
        if r['Price'] and float(r['Time']) >= tm:
            r['Price'] = str(ref + 2 * (int(r['Price']) - ref))
    return rows


def bigtaker(rows, k):
    for r in rows:
        if r['Operation'] == 'Insert' and r['Lifespan'] == 'F':
            r['Volume'] = str(max(1, round(k * int(r['Volume']))))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('inputs', type=Path, nargs='+')
    ap.add_argument('--out-dir', type=Path, default=ROOT / 'data' / 'stress')
    ap.add_argument('--levels', type=float, nargs='+', default=[1, 3])
    ap.add_argument('--only', nargs='+')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    for path in a.inputs:
        with path.open(newline='') as f:
            rows = list(csv.DictReader(f))
        t0, t1 = float(rows[0]['Time']), float(rows[-1]['Time'])
        tags = dict(A.DETERMINISTIC)
        for lv in a.levels:
            tags.update(A.noise_presets(lv))
        extra = ['bigtaker2', 'bigtaker4', 'burst', 'trendup', 'trenddown', 'volreg'] + \
            [f'toxic{lv:g}' for lv in a.levels]
        names = list(tags) + extra
        if a.only:
            names = [n for n in names if n in a.only]
        mid_at = None
        for tag in names:
            rng = random.Random(f'{a.seed}-{path.stem}-{tag}')
            rs = [dict(r) for r in rows]
            if tag in tags:
                rs = A.transform(rs, tags[tag], rng)
            elif tag.startswith('toxic'):
                mid_at = mid_at or future_mid(path)
                rs = toxic_steps(rs, 0.1 * float(tag[5:]), rng, mid_at)
            elif tag.startswith('bigtaker'):
                rs = bigtaker(rs, float(tag[8:]))
            elif tag == 'burst':
                rs = price_shift(rs, burst_fn(t0, t1))
            elif tag in ('trendup', 'trenddown'):
                sg = 1 if tag == 'trendup' else -1
                rs = price_shift(rs, lambda t: sg * 300 * (t - t0) / (t1 - t0))
            elif tag == 'volreg':
                rs = volreg(rs)
            out = a.out_dir / f'{path.stem}_{tag}.csv'
            with out.open('w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=A.HEADER, lineterminator='\n')
                w.writeheader()
                w.writerows(rs)
            print(f'{path.stem} {tag:12} rows={len(rs)} -> {out}', flush=True)


if __name__ == '__main__':
    main()
