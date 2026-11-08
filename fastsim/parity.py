"""Check that fast-sim reproduces exchange-simulator byte for byte (result.json and pnl.csv).

    uv run python -m fastsim.parity --record     # golden outputs from the reference simulator (slow, once)
    uv run python -m fastsim.parity              # fast engine vs golden, every case
    uv run python -m fastsim.parity --jit        # same, with the strategies' JIT_KERNELS compiled

Golden outputs live in data/.cache/fastsim_golden/<strategy>__<tape>/ (git-ignored, regenerable).
A case whose strategy or tape file is missing is skipped and reported.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / 'data/.cache/fastsim_golden'
STRATEGIES = [
    'strategies/marek/round3/r3_best.py',
    'strategies/marek/round3/r3_lad.py',
    'strategies/marek/round2/shockgen_best.py',
    'strategies/trend_follower.py',
]
TAPES = ['data/round3.csv', 'data/synth/r3_v1_seed1.csv']


def cases() -> list[tuple[str, str, Path]]:
    return [(s, t, GOLDEN / f'{Path(s).stem}__{Path(t).stem}') for s in STRATEGIES for t in TAPES]


def _record(case: tuple[str, str, Path]) -> str:
    strategy, tape, out = case
    p = subprocess.run([sys.executable, '-m', 'simulator.main', '--strategy', strategy, '--backtest', tape,
                        '--log-dir', str(out), '--no-logs'], cwd=ROOT, capture_output=True, text=True)
    return f'{out.name}: {"ok" if p.returncode == 0 else "FAILED " + p.stderr[-500:]}'


def _check(case: tuple[str, str, Path], jit: bool) -> tuple[str, bool, str]:
    from fastsim.engine import run_backtest, write_outputs

    strategy, tape, golden = case
    started = time.perf_counter()
    result, curve = run_backtest(ROOT / strategy, ROOT / tape, jit=jit)
    secs = time.perf_counter() - started
    with tempfile.TemporaryDirectory() as tmp:
        write_outputs(Path(tmp), result, curve)
        diffs = [name for name in ('result.json', 'pnl.csv')
                 if (Path(tmp) / name).read_bytes() != (golden / name).read_bytes()]
    note = f'pnl {result.pnl:,.2f}  {secs:.0f}s'
    return golden.name, not diffs, note + (f'  DIFFERS: {", ".join(diffs)}' if diffs else '')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--record', action='store_true', help='(re)write the golden outputs with exchange-simulator')
    ap.add_argument('--jit', action='store_true')
    ap.add_argument('--jobs', type=int, default=8)
    args = ap.parse_args()

    todo, skipped = [], []
    for case in cases():
        missing = [p for p in case[:2] if not (ROOT / p).exists()]
        if not missing and not args.record and not (case[2] / 'result.json').exists():
            missing = ['golden output (run with --record)']
        (skipped if missing else todo).append((case, missing))
    for case, missing in skipped:
        print(f'SKIP {case[2].name}: missing {", ".join(map(str, missing))}')

    with cf.ProcessPoolExecutor(min(args.jobs, len(todo) or 1)) as ex:
        if args.record:
            for line in ex.map(_record, [c for c, _ in todo]):
                print(line)
            return
        results = list(ex.map(_check, [c for c, _ in todo], [args.jit] * len(todo)))
    for name, ok, note in results:
        print(f'{"PASS" if ok else "FAIL"} {name:40s} {note}')
    failed = sum(not ok for _, ok, _ in results)
    print(f'{len(results) - failed}/{len(results)} identical' + (f', {len(skipped)} skipped' if skipped else ''))
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
