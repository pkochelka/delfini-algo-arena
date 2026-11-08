"""Paired evaluation of PARAMS variants against a base on the round-4 tape (fastsim, cached).

    uv run python strategies/peter/r4_eval.py STRAT.py '{"k":1}' '{"k":2}' [--base '{}'] [--tier wf|hold|full|r3] [--jobs 9]
    uv run python strategies/peter/r4_eval.py A.py --vs B.py --tier full     # compare two files

Tiers:
    wf    real round-4 wf01-12 (60 s warmup + ~370 s each, first 75 % of the tape): iterate here
    hold  the 4 hold16 folds that lie past 4500 s (never tuned on): confirm here
    full  the whole round-4 tape in one run, PnL per 500 s piece (12 pieces)
    r3    real round-3 wf01-12: a cross-round sanity check (the private r4 tape is another draw)
Each line: total delta, t-stat over windows/pieces, windows won, and the variant's own total.
Cache: logs/r4/cache.json keyed by sha1(source) + params + window.
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


def windows(tier: str) -> tuple[list[dict], float]:
    if tier == 'full':
        return [{'path': 'data/round4_public.csv', 'start': 0}], 500.0
    if tier == 'r3full':
        return [{'path': 'data/round3_public.csv', 'start': 0}], 500.0
    d = {'wf': 'data/splits_r4/wf', 'hold': 'data/splits_r4/hold16', 'r3': 'data/splits_r3/wf'}[tier]
    fs = json.loads((ROOT / d / 'folds.json').read_text())['folds']
    if tier == 'hold':
        fs = [f for f in fs if f['eval_start_s'] >= 4500]
    return [{'path': f'{d}/{f["file"]}', 'start': f['trade_start_us']} for f in fs], 0.0


def run_one(strategy: str, over: dict, w: dict, win_s: float) -> dict:
    sys.path.insert(0, str(ROOT))
    from fastsim.engine import grader_config, run_backtest
    params = over | ({'trade_start_us': w['start']} if w['start'] else {})
    try:
        r, curve = run_backtest(strategy, ROOT / w['path'], params, config=grader_config())
    except ValueError:  # strategy without PARAMS: no overrides possible
        r, curve = run_backtest(strategy, ROOT / w['path'], None, config=grader_config())
    parts = [0.0]
    if win_s:
        n = int(curve[-1][0] // win_s) + 1
        parts, prev = [0.0] * n, 0.0
        for s, v in curve:
            parts[min(int(s // win_s), n - 1)] += v - prev
            prev = v
    else:
        parts = [r.pnl]
    return {'pnl': r.pnl, 'parts': parts, 'breach': r.breach_penalty, 'fills': r.fills,
            'q95': r.abs_inventory_q95}


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('strategy')
    ap.add_argument('variants', nargs='*')
    ap.add_argument('--vs', action='append', default=[], help='other strategy files to compare against the base')
    ap.add_argument('--base', default='{}')
    ap.add_argument('--tier', default='wf', choices=['wf', 'hold', 'full', 'r3', 'r3full'])
    ap.add_argument('--jobs', type=int, default=9)
    ap.add_argument('--no-cache', action='store_true')
    a = ap.parse_args()
    ws, win_s = windows(a.tier)
    cache = {} if a.no_cache or not CACHE.exists() else json.loads(CACHE.read_text())
    base = json.loads(a.base)
    runs = [(a.strategy, {})] + [(a.strategy, json.loads(v)) for v in a.variants] + [(s, {}) for s in a.vs]

    def key(s, v, w):
        src = hashlib.sha1(Path(s).read_bytes()).hexdigest()[:12]
        return f"{src}|{json.dumps(base | v, sort_keys=True)}|{w['path']}|{w['start']}|{win_s}"

    keys = [[key(s, v, w) for w in ws] for s, v in runs]
    with ProcessPoolExecutor(a.jobs) as ex:
        futs = {}
        for (s, v), ks in zip(runs, keys):
            for w, k in zip(ws, ks):
                if k not in cache and k not in futs:
                    futs[k] = ex.submit(run_one, s, base | v, w, win_s)

        def get(k):
            if k not in cache:
                cache[k] = futs[k].result()
                save({k: cache[k]})
            return cache[k]

        b = [get(k) for k in keys[0]]
        tot = lambda rs: sum(r['pnl'] for r in rs) / 1e3  # noqa: E731
        print(f"[{a.tier}: {len(ws)} windows, {len(futs)} runs] {Path(a.strategy).name} {json.dumps(base)[:60]}: "
              f"{tot(b):.1f}k  parts {' '.join(f'{p / 1e3:.0f}' for r in b for p in r['parts'])}"
              f"  breach {sum(r['breach'] for r in b):.0f}", flush=True)
        for (s, v), ks in zip(runs[1:], keys[1:]):
            rs = [get(k) for k in ks]
            dw = [p - q for x, y in zip(rs, b) for p, q in zip(x['parts'], y['parts'])]
            name = json.dumps(v) if s == a.strategy else Path(s).name
            print(f"{name[:70]:70} d {sum(dw) / 1e3:+7.1f}k t {tstat(dw):+5.1f} won {sum(x > 0 for x in dw):2d}/{len(dw)}"
                  f" | {tot(rs):7.1f}k breach {sum(r['breach'] for r in rs):.0f}", flush=True)


if __name__ == '__main__':
    main()
