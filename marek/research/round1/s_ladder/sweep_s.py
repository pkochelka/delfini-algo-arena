"""A/B s_ladder variants on the round1 walk-forward folds wf01-10, paired per fold against a base.

    uv run python marek/research/round1/s_ladder/sweep_s.py --base '{"s_pull":0}' label '{"s_pull":0.45}' ...

Pairs of (label, json overrides). Results are appended to results.jsonl next to this file.
"""
from __future__ import annotations

import json
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'strategies' / 'peter'))
from tune import load_folds, run_fold  # noqa: E402

STRAT = str(ROOT / 'strategies' / 'marek' / 's_ladder.py')
FOLDS = load_folds(ROOT / 'data/splits_round1/wf', list(range(1, 11)))
OUT = Path(__file__).with_name('results.jsonl')


def run(name, ov, pool, base):
    res = list(pool.map(run_fold, [STRAT] * len(FOLDS), [ov] * len(FOLDS), FOLDS))
    pnls = [r['pnl'] for r in res]
    breach = sum(r['breach'] for r in res)
    line = {'name': name, 'params': ov, 'total': sum(pnls), 'folds': pnls, 'breach': breach}
    msg = f"{name:<30} total={sum(pnls):>10,.0f}"
    if base:
        d = [a - b for a, b in zip(pnls, base)]
        sd = statistics.stdev(d)
        line |= {'delta': sum(d), 'sd': sd}
        msg += f"  delta={sum(d):>+9,.0f}  sd={sd:>6,.0f}  t={statistics.fmean(d) / (sd / len(d) ** .5 or 1):+5.1f}  wins={sum(x > 0 for x in d)}/{len(d)}"
    if breach:
        msg += f"  BREACH={breach:,.0f}"
    print(msg, flush=True)
    with OUT.open('a') as f:
        f.write(json.dumps(line) + '\n')
    return pnls


def main():
    a = sys.argv[1:]
    global FOLDS
    base_ov = {}
    if a[:1] == ['--synth']:  # validate on synthetic round1-generator tapes instead (data/synth/r1_val_seed*.csv)
        FOLDS = [{'name': f.stem, 'path': str(f), 'trade_start_us': 0} for f in sorted((ROOT / 'data/synth').glob('r1_val_seed*.csv'))]
        a = a[1:]
    if a[:1] == ['--base']:
        base_ov = json.loads(a[1]); a = a[2:]
    todo = [(a[i], json.loads(a[i + 1])) for i in range(0, len(a), 2)]
    with ProcessPoolExecutor(max_workers=10) as pool:
        base = run('base ' + json.dumps(base_ov), base_ov, pool, None)
        for name, ov in todo:
            run(name, base_ov | ov, pool, base)


if __name__ == '__main__':
    main()
