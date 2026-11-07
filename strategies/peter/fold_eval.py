"""Parallel eval: uv run python strategies/peter/fold_eval.py strategy.py '{"a":1}' '{"a":2}' ... [--folds 1-12] [--files a.csv,b.csv]"""
import argparse, json, statistics, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
sys.path.insert(0, 'strategies/peter')
from tune import run_fold, load_folds, parse_range

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('strategy'); ap.add_argument('variants', nargs='*')
    ap.add_argument('--folds', default='1-12'); ap.add_argument('--dir', default='data/splits_r1/wf')
    ap.add_argument('--files', default=''); ap.add_argument('--jobs', type=int, default=12)
    a = ap.parse_args()
    if a.files:
        folds = [{'name': Path(p).stem[:14], 'path': p, 'trade_start_us': 0} for p in a.files.split(',')]
    else:
        folds = load_folds(Path(a.dir), parse_range(a.folds))
    vs = [json.loads(v) for v in a.variants] or [{}]
    with ProcessPoolExecutor(a.jobs) as ex:
        futs = [[ex.submit(run_fold, a.strategy, v, f) for f in folds] for v in vs]
        for v, fs in zip(vs, futs):
            res = [x.result() for x in fs]; p = [r['pnl'] for r in res]
            sd = statistics.pstdev(p) if len(p) > 1 else 0
            print(f"{json.dumps(v):70} mean {statistics.fmean(p):+9.0f} sd {sd:7.0f} min {min(p):+8.0f} breach {sum(r['breach'] for r in res):.0f} q95 {statistics.fmean(r['q95'] for r in res):.0f}")
            print('    ' + ' '.join(f"{r['fold']}={r['pnl']:+.0f}" for r in res), flush=True)

if __name__ == '__main__':
    main()
