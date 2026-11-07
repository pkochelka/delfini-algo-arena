"""A/B sig_ladder combinations (guard + touch/behind split + improve_both) on wf01-10.

    uv run python marek/research/round1/combo/sweep_combo.py NAME [NAME ...]
    uv run python marek/research/round1/combo/sweep_combo.py --json '{"size_deep":100}' label

Deltas are paired per fold against sig_ladder defaults. Results are appended to results.jsonl.
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

STRAT = str(ROOT / 'strategies' / 'marek' / 'sig_ladder.py')
FOLDS = load_folds(ROOT / 'data/splits_round1/wf', list(range(1, 11)))
OUT = Path(__file__).with_name('results.jsonl')

VARIANTS: dict[str, dict] = {
    'base': {},
    'split 100/100': {'size_top': 100, 'size_deep': 100},
    'split 100/100 pjtop': {'size_top': 100, 'size_deep': 100, 'pj_topup': 1},
    'split 150/50': {'size_top': 150, 'size_deep': 50},
    'split 200/100': {'size_top': 200, 'size_deep': 100},
    'improve_both': {'improve_both': 1},
    'split 100/100 ib': {'size_top': 100, 'size_deep': 100, 'improve_both': 1},
    # short-window flow term
    'f2 .5': {'flow2_k': 0.5},
    'f2 1': {'flow2_k': 1.0},
    'f2 2': {'flow2_k': 2.0},
    'f2 4': {'flow2_k': 4.0},
    'f2 1 50ms': {'flow2_k': 1.0, 'flow2_ms': 50},
    'f2 1 200ms': {'flow2_k': 1.0, 'flow2_ms': 200},
    'f2 .5 200ms': {'flow2_k': 0.5, 'flow2_ms': 200},
    'f2 0.25 200ms': {'flow2_k': 0.25, 'flow2_ms': 200},
    'f2 0.25 300ms': {'flow2_k': 0.25, 'flow2_ms': 300},
    'f2 0.25 500ms': {'flow2_k': 0.25, 'flow2_ms': 500},
    'f2 0.5 200ms': {'flow2_k': 0.5, 'flow2_ms': 200},
    'f2 0.5 300ms': {'flow2_k': 0.5, 'flow2_ms': 300},
    'f2 0.5 500ms': {'flow2_k': 0.5, 'flow2_ms': 500},
    'f2 0.75 200ms': {'flow2_k': 0.75, 'flow2_ms': 200},
    'f2 0.75 300ms': {'flow2_k': 0.75, 'flow2_ms': 300},
    'f2 0.75 500ms': {'flow2_k': 0.75, 'flow2_ms': 500},
    'f2 .5 300ms ib': {'flow2_k': 0.5, 'flow2_ms': 300, 'improve_both': 1},
    'f2 .5 200ms ib': {'flow2_k': 0.5, 'flow2_ms': 200, 'improve_both': 1},
    'f2 .5 200ms fk0': {'flow2_k': 0.5, 'flow2_ms': 200, 'flow_k': 0.0},
    # severe tier: also pull the flattening part
    'sev .03/.2': {'sev_share': 0.03, 'sev_dep': 0.2},
    'sev .05/.3': {'sev_share': 0.05, 'sev_dep': 0.3},
    'sev .02/.1': {'sev_share': 0.02, 'sev_dep': 0.1},
}


def run(name: str, ov: dict, pool, base: list[float] | None) -> list[float]:
    pnls = [r['pnl'] for r in pool.map(run_fold, [STRAT] * len(FOLDS), [ov] * len(FOLDS), FOLDS)]
    line = {'name': name, 'params': ov, 'total': sum(pnls), 'folds': pnls}
    msg = f"{name:<28} total={sum(pnls):>10,.0f}"
    if base:
        d = [a - b for a, b in zip(pnls, base)]
        sd = statistics.stdev(d)
        line |= {'delta': sum(d), 'sd': sd}
        msg += (f"  delta={sum(d):>+9,.0f}  fold sd={sd:>6,.0f}  t={statistics.fmean(d) / (sd / 10 ** .5 or 1):+5.1f}"
                f"  wins={sum(x > 0 for x in d)}/10")
    print(msg, flush=True)
    with OUT.open('a') as f:
        f.write(json.dumps(line) + '\n')
    return pnls


def main() -> None:
    args = sys.argv[1:]
    todo = [(args[2], json.loads(args[1]))] if args[:1] == ['--json'] else [(n, VARIANTS[n]) for n in (args or VARIANTS)]
    with ProcessPoolExecutor(max_workers=10) as pool:
        base = run('base', {}, pool, None)
        for name, ov in todo:
            if name != 'base':
                run(name, ov, pool, base)


if __name__ == '__main__':
    main()
