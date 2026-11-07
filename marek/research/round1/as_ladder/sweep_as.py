"""Fold A/B for strategies/marek/as_ladder.py (A-S overlay on sig_ladder) on the round1 walk-forward folds.

    uv run python marek/research/round1/as_ladder/sweep_as.py [variant ...]

Prints per variant the fold total, the paired difference against the 'base' variant (all as_* off = sig_ladder),
its t-stat and win count, plus fills, mean |inv| q95 and breach penalty. Appends rows to results.jsonl.
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

STRAT = str(ROOT / 'strategies' / 'marek' / 'as_ladder.py')
FOLDS = load_folds(ROOT / 'data/splits_round1/wf', list(range(1, 11)))
OUT = Path(__file__).with_name('results.jsonl')

G = {'as_gamma': 1e-4, 'as_tau_s': 100.0}  # skew ~0.85 tick at q = 100 on round1
VARIANTS: dict[str, dict] = {
    'base': {},
    # exponential sizing of the adding side
    'size k1': {'as_size_k': 1.0},
    'size k2': {'as_size_k': 2.0},
    'size k4': {'as_size_k': 4.0},
    'size k2 shrink': {'as_size_k': 2.0, 'as_shrink': 1},
    'size k4 shrink': {'as_size_k': 4.0, 'as_shrink': 1},
    # back off the adding side
    'back 0.5': {'as_back': 0.5},
    'back 0.8': {'as_back': 0.8},
    # step the reducing side inside wide spreads
    'improve': {'as_improve': 1, 'as_gamma': 2e-4},
    # active IOC unwind
    'take 0.6': {'as_take': 0.6},
    'take 0.8': {'as_take': 0.8},
    'take 0.8 l20': {'as_take': 0.8, 'as_take_lots': 20},
}


def main() -> None:
    sel = sys.argv[1:] or list(VARIANTS)
    if 'base' not in sel:
        sel = ['base'] + sel
    with ProcessPoolExecutor() as pool:
        jobs = {n: [pool.submit(run_fold, STRAT, G | VARIANTS[n], f) for f in FOLDS] for n in sel}
        base = None
        for name, futs in jobs.items():
            res = [f.result() for f in futs]
            pnls = [r['pnl'] for r in res]
            if base is None:
                base = pnls
            d = [a - b for a, b in zip(pnls, base)]
            sd = statistics.stdev(d) if len(d) > 1 else 0.0
            t = statistics.fmean(d) / (sd / len(d) ** 0.5) if sd else 0.0
            row = {'variant': name, 'params': G | VARIANTS[name], 'total': sum(pnls), 'delta': sum(d), 't': t,
                   'wins': sum(x > 0 for x in d), 'fills': sum(r['fills'] for r in res),
                   'q95': statistics.fmean(r['q95'] for r in res), 'breach': sum(r['breach'] for r in res),
                   'folds': pnls}
            with OUT.open('a') as fh:
                fh.write(json.dumps(row) + '\n')
            print(f"{name:<18} total={row['total']:>9.0f} d={row['delta']:>+8.0f} t={t:>+5.1f} "
                  f"wins={row['wins']:>2} fills={row['fills']:>6} q95={row['q95']:5.1f} breach={row['breach']:.0f}",
                  flush=True)


if __name__ == '__main__':
    main()
