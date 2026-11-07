"""as_ladder variants on the trending / thin tapes where the ladder loses (FINDINGS: Robustness).

    uv run python marek/research/round1/as_ladder/stress_as.py
"""
from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'strategies' / 'peter'))
sys.path.insert(0, str(Path(__file__).parent))
from sweep_as import G, STRAT, VARIANTS  # noqa: E402
from tune import run_fold  # noqa: E402

TAPES = ['data/random_stress.csv', 'data/sample.csv', 'data/sample_v2.csv', 'data/stress/wf03_trendup.csv',
         'data/stress/wf03_toxic10.csv', 'data/stress/wf07_vscale0.3.csv', 'data/stress/wf07_drift100.csv']


def main() -> None:
    sel = sys.argv[1:] or list(VARIANTS)
    tapes = [{'name': Path(t).stem, 'path': str(ROOT / t), 'trade_start_us': 0} for t in TAPES]
    with ProcessPoolExecutor() as pool:
        jobs = {n: [pool.submit(run_fold, STRAT, G | VARIANTS[n], t) for t in tapes] for n in sel}
        print(f"{'variant':<18}" + ''.join(f'{t["name"][:14]:>15}' for t in tapes))
        for n, futs in jobs.items():
            print(f'{n:<18}' + ''.join(f'{f.result()["pnl"]:>15.0f}' for f in futs), flush=True)


if __name__ == '__main__':
    main()
