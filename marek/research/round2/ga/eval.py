"""GA fitness evaluation for round 2 strategies.

usage: uv run python marek/research/round2/ga/eval.py <strategy.py> --id ID [--parent P] [--desc "..."] [--set train|holdout|all]

Runs the strategy on every tape of the set (in parallel, but at most SLOTS simulations machine-wide across all
callers, via file locks), prints a JSON summary and appends it to marek/research/round2/ga/ledger.jsonl.
fitness = mean PnL over the set; any breach or crash -> fitness = -1e9.
"""
import argparse, fcntl, json, os, pathlib, shutil, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor

ROOT = pathlib.Path(__file__).resolve().parents[4]
GA = pathlib.Path(__file__).resolve().parent
SLOTS = int(os.environ.get('GA_SLOTS', '8'))
SETS = {
    'train': ['data/round2.csv'] + [f'data/synth/r2_v2_seed{k}.csv' for k in (1, 2, 3, 4)],
    'holdout': [f'data/synth/r2_v2_seed{k}.csv' for k in (5, 6, 7)] + [f'data/synth/r2_v1_seed{k}.csv' for k in (1, 2, 3)],
}
SETS['all'] = SETS['train'] + SETS['holdout']


def acquire():
    (GA / 'locks').mkdir(exist_ok=True)
    while True:
        for i in range(SLOTS):
            f = open(GA / 'locks' / f'slot{i}', 'w')
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return f
            except BlockingIOError:
                f.close()
        time.sleep(0.5)


def run_one(src: pathlib.Path, ident: str, tape: str):
    d = ROOT / 'marek/logs/ga' / ident / pathlib.Path(tape).stem
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    strat = d / 'strategy.py'
    shutil.copy(src, strat)
    lock = acquire()
    try:
        t0 = time.time()
        out = subprocess.run(['uv', 'run', 'exchange-simulator', '--strategy', str(strat), '--backtest', str(ROOT / tape),
                              '--no-logs', '--log-dir', str(d)], capture_output=True, text=True, cwd=ROOT, timeout=1800)
    finally:
        lock.close()
    (d / 'stdout.txt').write_text(out.stdout + out.stderr)
    rf = d / 'result.json'
    if not rf.exists():
        return {'tape': tape, 'pnl': None, 'error': (out.stdout + out.stderr)[-1500:]}
    r = json.loads(rf.read_text())
    return {'tape': tape, 'pnl': r['pnl'], 'breach': r.get('breach_penalty', 0), 'fills': r.get('fills'),
            'secs': round(time.time() - t0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('strategy')
    ap.add_argument('--id', required=True)
    ap.add_argument('--parent', default='')
    ap.add_argument('--desc', default='')
    ap.add_argument('--set', default='train', choices=list(SETS))
    a = ap.parse_args()
    src = pathlib.Path(a.strategy).resolve()
    # quick syntax/import check before burning simulator time
    chk = subprocess.run(['uv', 'run', 'python', '-c', f'import runpy; runpy.run_path({str(src)!r})'],
                         capture_output=True, text=True, cwd=ROOT)
    if chk.returncode != 0:
        rec = {'id': a.id, 'parent': a.parent, 'desc': a.desc, 'set': a.set, 'fitness': -1e9,
               'error': 'import failed: ' + chk.stderr[-1500:]}
    else:
        with ThreadPoolExecutor(len(SETS[a.set])) as ex:
            res = list(ex.map(lambda t: run_one(src, a.id, t), SETS[a.set]))
        bad = [r for r in res if r['pnl'] is None or (r.get('breach') or 0) > 0]
        pn = {pathlib.Path(r['tape']).stem: r['pnl'] for r in res}
        vals = [r['pnl'] for r in res if r['pnl'] is not None]
        fit = -1e9 if bad else sum(vals) / len(vals)
        rec = {'id': a.id, 'parent': a.parent, 'desc': a.desc, 'set': a.set, 'file': str(src.relative_to(ROOT)),
               'fitness': round(fit), 'min': round(min(vals)) if vals else None, 'pnl': pn}
        if bad:
            rec['error'] = [{k: r.get(k) for k in ('tape', 'breach', 'error')} for r in bad]
    with open(GA / 'ledger.jsonl', 'a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(rec) + '\n')
    print(json.dumps(rec, indent=1))


if __name__ == '__main__':
    main()
