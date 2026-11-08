"""Paired evaluation of PARAMS variants against a base: ONE run of the round-4 tape per model (fastsim, cached).

    uv run python strategies/peter/r4_eval.py STRAT.py '{"k":1}' '{"k":2}' [--base '{}'] [--jobs 9]
    uv run python strategies/peter/r4_eval.py A.py --vs B.py                  # compare two files
    uv run python strategies/peter/r4_eval.py STRAT.py '{"k":1}' --tape data/round3_public.csv   # another tape

Every model (strategy file + params) trades the whole tape once; all statistics are cut from that run's PnL
curve (stored in 50 s bins), so nothing is re-run per fold:
    total   whole-tape PnL
    dev     PnL before 4500 s (the part the round-4 audit / splits treat as development)
    hold    PnL from 4500 s on (never used for selection in splits_r4)
    t, won  over the 12 500 s pieces of the paired delta
Variants run in parallel, one process each (~4 min for a dev_best-sized strategy on round 4).
Selection caveat: one public tape overfits (lamj_v/bj_v: +27k here, -36k on fresh synth). Confirm candidates on
session db's fresh round-4 synth tapes: strategies/peter/r4_full.py STRAT.py '{...}' --syn sel.
Cache: logs/r4/cache.json keyed by sha1(source) + params + tape.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
CACHE = ROOT / 'logs' / 'r4' / 'cache.json'
BIN_S = 50.0
PIECE_S = 500.0
HOLD_S = 4500.0
VER = 'one1'


def run_one(strategy: str, over: dict, tape: str) -> dict:
    sys.path.insert(0, str(ROOT))
    from fastsim.engine import grader_config, run_backtest
    try:
        r, curve = run_backtest(strategy, ROOT / tape, over, config=grader_config())
    except ValueError:  # strategy without PARAMS: no overrides possible
        r, curve = run_backtest(strategy, ROOT / tape, None, config=grader_config())
    n = int(curve[-1][0] // BIN_S) + 1
    bins, prev = [0.0] * n, 0.0
    for s, v in curve:
        bins[min(int(s // BIN_S), n - 1)] += v - prev
        prev = v
    return {'pnl': r.pnl, 'bins': bins, 'breach': r.breach_penalty, 'fills': r.fills}


def save(new: dict) -> None:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    disk = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    disk.update(new)
    tmp = CACHE.with_suffix(f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(disk))
    for k in range(50):  # Windows: the target may be open in another process for a moment
        try:
            os.replace(tmp, CACHE)
            return
        except PermissionError:
            time.sleep(0.1 * (k + 1))


def tstat(d: list[float]) -> float:
    sd = statistics.stdev(d) if len(d) > 1 else 0.0
    return statistics.fmean(d) / (sd / math.sqrt(len(d))) if sd else 0.0


def split(r: dict) -> tuple[float, float, list[float]]:
    b = r['bins']
    k = int(HOLD_S // BIN_S)
    per = int(PIECE_S // BIN_S)
    pieces = [sum(b[i:i + per]) for i in range(0, len(b), per)]
    return sum(b[:k]), sum(b[k:]), pieces


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('strategy')
    ap.add_argument('variants', nargs='*')
    ap.add_argument('--vs', action='append', default=[], help='other strategy files to compare against the base')
    ap.add_argument('--base', default='{}')
    ap.add_argument('--tape', default='data/round4_public.csv')
    ap.add_argument('--jobs', type=int, default=9)
    ap.add_argument('--no-cache', action='store_true')
    a = ap.parse_args()
    cache = {} if a.no_cache or not CACHE.exists() else json.loads(CACHE.read_text())
    base = json.loads(a.base)
    runs = [(a.strategy, {})] + [(a.strategy, json.loads(v)) for v in a.variants] + [(s, {}) for s in a.vs]

    def key(s, v):
        src = hashlib.sha1(Path(s).read_bytes()).hexdigest()[:12]
        return f"{src}|{json.dumps(base | v, sort_keys=True)}|{a.tape}|{VER}"

    keys = [key(s, v) for s, v in runs]
    with ProcessPoolExecutor(a.jobs) as ex:
        futs = {}
        for (s, v), k in zip(runs, keys):
            if k not in cache and k not in futs:
                futs[k] = ex.submit(run_one, s, base | v, a.tape)

        def get(k):
            if k not in cache:
                cache[k] = futs[k].result()
                save({k: cache[k]})
            return cache[k]

        b = get(keys[0])
        bd, bh, bp = split(b)
        print(f"[{Path(a.tape).stem}: {len(futs)} runs] {Path(a.strategy).name} {json.dumps(base)[:60]}: "
              f"total {b['pnl'] / 1e3:.1f}k  dev {bd / 1e3:.1f}k  hold {bh / 1e3:.1f}k  breach {b['breach']:.0f}"
              f"  pieces {' '.join(f'{p / 1e3:.0f}' for p in bp)}", flush=True)
        for (s, v), k in zip(runs[1:], keys[1:]):
            r = get(k)
            d, h, ps = split(r)
            dw = [p - q for p, q in zip(ps, bp)]
            name = json.dumps(v) if s == a.strategy else Path(s).name
            print(f"{name[:66]:66} d {(r['pnl'] - b['pnl']) / 1e3:+7.1f}k t {tstat(dw):+5.1f} won {sum(x > 0 for x in dw):2d}/{len(dw)}"
                  f" | dev {(d - bd) / 1e3:+6.1f}k hold {(h - bh) / 1e3:+6.1f}k | {r['pnl'] / 1e3:7.1f}k", flush=True)


if __name__ == '__main__':
    main()
