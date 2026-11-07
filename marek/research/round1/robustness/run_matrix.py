"""Run strategies x tapes through the simulator (--no-logs), at most N in parallel, and tabulate.

Usage:
    uv run python research/round1/robustness/run_matrix.py --out research/round1/robustness/results/base.csv \
        --strategy ladder=strategies/peter/ladder_mm.py \
        --strategy 'ladder_cap50=strategies/peter/ladder_mm.py:{"cap": 50}' \
        --tape data/sample_v2.csv --tape data/stress/*.csv --jobs 3

A `path:{json}` strategy gets a temporary copy with `PARAMS.update(json)` appended (the simulator
loads one file, and every strategy here reads the module PARAMS at construction).
Results land in logs/robust/<strategy>/<tape>/ and one row per run in --out (appended, deduped).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TMP = Path(tempfile.gettempdir()) / 'robust_strats'


def materialise(spec: str) -> tuple[str, Path]:
    name, rest = spec.split('=', 1)
    if ':{' in rest:
        path, js = rest.split(':', 1)
        over = json.loads(js)
        TMP.mkdir(parents=True, exist_ok=True)
        src = (ROOT / path).read_text()
        h = hashlib.md5((path + js).encode()).hexdigest()[:8]
        out = TMP / f'{Path(path).stem}_{h}.py'
        out.write_text(src + f'\n\nPARAMS.update({over!r})\n')
        return name, out
    return name, ROOT / rest


def max_dd(pnl_csv: Path) -> tuple[float, float]:
    peak, dd, last = 0.0, 0.0, 0.0
    with pnl_csv.open() as f:
        for row in csv.DictReader(f):
            v = float(row['pnl'])
            peak = max(peak, v)
            dd = max(dd, peak - v)
            last = v
    return dd, last


def run(name: str, path: Path, tape: Path) -> dict:
    log = ROOT / 'logs' / 'robust' / name / tape.stem
    cmd = ['uv', 'run', 'exchange-simulator', '--strategy', str(path), '--backtest', str(tape),
           '--no-logs', '--log-dir', str(log)]
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    row = {'strategy': name, 'tape': tape.stem}
    if p.returncode != 0:
        row['error'] = p.stderr[-300:].replace('\n', ' ')
        return row
    res = json.loads((log / 'result.json').read_text())
    dd, _ = max_dd(log / 'pnl.csv')
    row.update({k: res[k] for k in ('pnl', 'breach_penalty', 'fills', 'traded_volume', 'inventory',
                                    'abs_inventory_q50', 'abs_inventory_q95', 'pnl_se')})
    row['max_dd'] = dd
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--strategy', action='append', required=True)
    ap.add_argument('--tape', nargs='+', action='extend', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--jobs', type=int, default=3)
    a = ap.parse_args()
    strats = [materialise(s) for s in a.strategy]
    jobs = [(n, p, Path(t)) for n, p in strats for t in a.tape]
    with ThreadPoolExecutor(a.jobs) as ex:
        rows = list(ex.map(lambda j: run(*j), jobs))
    old = []
    if a.out.exists():
        with a.out.open() as f:
            old = [r for r in csv.DictReader(f)]
    keys = {(r['strategy'], r['tape']) for r in rows}
    allrows = [r for r in old if (r['strategy'], r['tape']) not in keys] + rows
    fields = ['strategy', 'tape', 'pnl', 'max_dd', 'breach_penalty', 'fills', 'traded_volume', 'inventory',
              'abs_inventory_q50', 'abs_inventory_q95', 'pnl_se', 'error']
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(allrows)
    for r in rows:
        print(f"{r['strategy']:24} {r['tape']:28} pnl={r.get('pnl')!s:>10} dd={r.get('max_dd')!s:>8} "
              f"pen={r.get('breach_penalty')} q50={r.get('abs_inventory_q50')} {r.get('error', '')}")


if __name__ == '__main__':
    main()
