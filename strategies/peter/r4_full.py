"""Paired full-tape evaluation with a shock / non-shock split (round 4).

    uv run python strategies/peter/r4_full.py STRAT.py '{"k":1}' '{"k":2}' [--base '{}'] [--tape data/round4_public.csv]

Runs the whole tape once per variant (cached in logs/r4/full_cache.json by source hash + params + tape) and splits
each paired PnL delta into the shock windows [outage start - 1 s, + 20 s) of the tape's 28 maker outages (found as in
regime_split.py) and the rest (`ex`). `ex` is what a filter / quoting change should be judged on: the shock sniper's
PnL is chaotic (one shock can swing +-20k). The t-stat is over the 12 500 s pieces of `ex`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / 'logs' / 'r4' / 'full_cache.json'


def shock_mask(tape: str, n: int) -> np.ndarray:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from regime_split import labels
    _, _, outages = labels(str(ROOT / tape), 5, 110)
    m = np.zeros(n, dtype=bool)
    for s, _ in outages:
        m[max(0, int(s) - 1):min(n, int(s) + 20)] = True
    return m


def run_one(strategy: str, over: dict, tape: str) -> list[float]:
    sys.path.insert(0, str(ROOT))
    from fastsim.engine import run_backtest
    _, curve = run_backtest(ROOT / strategy, ROOT / tape, over or None)
    n = int(curve[-1][0]) + 1
    per = [0.0] * n
    prev = 0.0
    for s, v in curve:
        per[min(int(s), n - 1)] += v - prev
        prev = v
    return per


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('strategy')
    ap.add_argument('variants', nargs='*')
    ap.add_argument('--base', default='{}')
    ap.add_argument('--tape', default='data/round4_public.csv')
    ap.add_argument('--jobs', type=int, default=8)
    a = ap.parse_args()
    base = json.loads(a.base)
    vs = [{}] + [json.loads(v) for v in a.variants]
    src = hashlib.sha1((ROOT / a.strategy).read_bytes()).hexdigest()[:12]
    keys = [f'{src}|{json.dumps(base | v, sort_keys=True)}|{a.tape}' for v in vs]
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    with ProcessPoolExecutor(a.jobs) as ex:
        futs = {k: ex.submit(run_one, a.strategy, base | v, a.tape) for k, v in zip(keys, vs) if k not in cache}
        for k, f in futs.items():
            cache[k] = f.result()
            disk = json.loads(CACHE.read_text()) if CACHE.exists() else {}
            disk[k] = cache[k]
            CACHE.parent.mkdir(parents=True, exist_ok=True)
            tmp = CACHE.with_suffix(f'.{os.getpid()}.tmp')
            tmp.write_text(json.dumps(disk))
            os.replace(tmp, CACHE)
    curves = [np.array(cache[k]) for k in keys]
    n = min(len(c) for c in curves)
    sm = shock_mask(a.tape, n)
    b = curves[0][:n]
    print(f'{Path(a.strategy).name} {json.dumps(base)[:50]} on {a.tape}: total {b.sum() / 1e3:.1f}k  '
          f'shock {b[sm].sum() / 1e3:.1f}k  ex {b[~sm].sum() / 1e3:.1f}k  ({sm.sum()} shock s)')
    for v, c in zip(vs[1:], curves[1:]):
        d = c[:n] - b
        dx = np.where(sm, 0.0, d)
        parts = [dx[i:i + 500].sum() for i in range(0, n, 500)]
        sd = statistics.stdev(parts)
        t = statistics.fmean(parts) / (sd / math.sqrt(len(parts))) if sd else 0.0
        print(f'{json.dumps(v)[:62]:62} d {d.sum() / 1e3:+7.1f}k  shock {d[sm].sum() / 1e3:+6.1f}k  '
              f'ex {dx.sum() / 1e3:+6.1f}k t {t:+4.1f} won {sum(p > 0 for p in parts)}/{len(parts)}  | {c[:n].sum() / 1e3:.1f}k')


if __name__ == '__main__':
    main()
