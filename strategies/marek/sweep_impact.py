"""Grid A/B for impact_mm on the round1 walk-forward folds (data/splits_round1/wf).

    uv run python strategies/marek/sweep_impact.py            # prints one line per variant
"""
from __future__ import annotations

import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'strategies' / 'peter'))
from tune import load_folds, run_fold  # noqa: E402

STRAT = str(ROOT / 'strategies' / 'marek' / 'impact_mm.py')
FOLDS = load_folds(ROOT / 'data/splits_round1/wf', list(range(1, 11)))

VARIANTS: dict[str, dict] = {
    'cap30 no guard': {'guard': 0, 'cap': 30},
    'cap30 pull600/t300': {'cap': 30},
    'cap30 pull300/t100': {'cap': 30, 'guard_lots': 300, 'tau_ms': 100},
    'cap30 freeze600': {'cap': 30, 'guard_mode': 'freeze'},
    'cap60 no guard': {'guard': 0, 'cap': 60},
    'cap60 pull600/t300': {'cap': 60},
    'cap60 pull300/t100': {'cap': 60, 'guard_lots': 300, 'tau_ms': 100},
    'cap60 freeze600': {'cap': 60, 'guard_mode': 'freeze'},
    'cap100 pull min800 /t300': {'min_order': 800, 'guard_lots': 800},
    'cap100 pull min1600/t500': {'min_order': 1600, 'guard_lots': 1600, 'tau_ms': 500},
}


def main() -> None:
    sel = sys.argv[1:] or list(VARIANTS)
    with ProcessPoolExecutor() as pool:
        jobs = {name: [pool.submit(run_fold, STRAT, VARIANTS[name], f) for f in FOLDS] for name in sel}
        for name, futs in jobs.items():
            res = [f.result() for f in futs]
            pnls = [r['pnl'] for r in res]
            print(f"{name:<20} total={sum(pnls):>9.0f} min={min(pnls):>8.0f} sd={statistics.pstdev(pnls):>7.0f} "
                  f"fills={sum(r['fills'] for r in res):>6} q95={statistics.fmean(r['q95'] for r in res):5.1f} "
                  f"breach={sum(r['breach'] for r in res):.0f}", flush=True)


if __name__ == '__main__':
    main()
