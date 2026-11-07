"""A/B variants on the round1 walk-forward folds (data/splits_round1/wf), paired against a baseline.

    uv run python research/round1/queue/ab.py base 'q2:strategies/marek/q_ladder.py:{"levels":2}' ...

A variant is `name` (a key of VARIANTS below) or `name:path:json-overrides`. The first variant is the
baseline; every other line also gets the paired fold difference (mean, sd, t = mean/(sd/sqrt(n)) of
the per-fold deltas), which is the honest noise yardstick for A/B on the same folds. Results are
appended as JSON lines to research/round1/queue/ab_results.jsonl. At most --jobs (3) worker processes.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'strategies' / 'peter'))
from tune import load_folds, run_fold  # noqa: E402

LADDER = str(ROOT / 'strategies' / 'peter' / 'ladder_mm.py')
QL = str(ROOT / 'strategies' / 'marek' / 'q_ladder.py')
VARIANTS: dict[str, tuple[str, dict]] = {
    # ladder_mm.py on disk is edited by other agents (cap was set to 1000 mid-run once), so the baseline is
    # q_ladder.py with default PARAMS, which is the committed ladder_mm (same fold PnLs, 425,118)
    'base': (QL, {}),
    'ladder': (LADDER, {'cap': 100}),
}
OUT = Path(__file__).resolve().parent / 'ab_results.jsonl'


def parse(spec: str) -> tuple[str, str, dict]:
    if spec in VARIANTS:
        path, ov = VARIANTS[spec]
        return spec, path, ov
    name, path, js = spec.split(':', 2)
    ov = {'cap': 100} | json.loads(js)  # pin the cap: never inherit an edited default
    return name, str(ROOT / path) if not path.startswith('/') else path, ov


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('variants', nargs='+')
    ap.add_argument('--folds', default='1-10')
    ap.add_argument('--jobs', type=int, default=3)
    args = ap.parse_args()
    a, _, b = args.folds.partition('-')
    idx = list(range(int(a), int(b or a) + 1))
    folds = load_folds(ROOT / 'data/splits_round1/wf', idx)
    specs = [parse(s) for s in args.variants]
    base_pnls = None
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        jobs = [(n, p, ov, [pool.submit(run_fold, p, ov, f) for f in folds]) for n, p, ov in specs]
        for name, path, ov, futs in jobs:
            res = [f.result() for f in futs]
            pnls = [r['pnl'] for r in res]
            line = (f"{name:<28} total={sum(pnls):>9.0f} min={min(pnls):>8.0f} sd={statistics.pstdev(pnls):>6.0f} "
                    f"fills={sum(r['fills'] for r in res):>6} q95={statistics.fmean(r['q95'] for r in res):5.1f} "
                    f"breach={sum(r['breach'] for r in res):.0f}")
            if base_pnls is None:
                base_pnls = pnls
            else:
                d = [x - y for x, y in zip(pnls, base_pnls)]
                sd = statistics.stdev(d) if len(d) > 1 else 0.0
                t = statistics.fmean(d) / (sd / math.sqrt(len(d))) if sd else 0.0
                line += f"  dTot={sum(d):>+8.0f} dSd={sd:>6.0f} t={t:+5.2f} wins={sum(x > 0 for x in d)}/{len(d)}"
            print(line, flush=True)
            with OUT.open('a') as fh:
                fh.write(json.dumps({'name': name, 'path': Path(path).name, 'overrides': ov,
                                     'folds': [r['fold'] for r in res], 'pnl': pnls,
                                     'fills': [r['fills'] for r in res], 'breach': [r['breach'] for r in res]}) + '\n')


if __name__ == '__main__':
    main()
