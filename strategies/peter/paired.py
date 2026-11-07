"""Paired hypothesis test on folds: each variant vs the base, per-fold difference (mean, folds won, t).

uv run python strategies/peter/paired.py strategy.py '{base}' '{"flag":1}' '{"flag":2}' ... [--dir data/splits_r2/wf] [--folds 1-12]
Each variant is applied on top of the base. Prints one line per variant, sorted as given.
"""
import argparse, json, math, statistics, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
sys.path.insert(0, 'strategies/peter')
from tune import run_fold, load_folds, parse_range


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('strategy'); ap.add_argument('base'); ap.add_argument('variants', nargs='*')
    ap.add_argument('--folds', default='1-12'); ap.add_argument('--dir', default='data/splits_r2/wf')
    ap.add_argument('--jobs', type=int, default=12)
    a = ap.parse_args()
    folds = load_folds(Path(a.dir), parse_range(a.folds))
    base = json.loads(a.base)
    vs = [{}] + [json.loads(v) for v in a.variants]
    with ProcessPoolExecutor(a.jobs) as ex:
        futs = [[ex.submit(run_fold, a.strategy, base | v, f) for f in folds] for v in vs]
        res = [[x.result() for x in fs] for fs in futs]
    b = [r['pnl'] for r in res[0]]
    print(f"base {json.dumps(base)}: mean {statistics.fmean(b):+.0f} min {min(b):+.0f} sum {sum(b):+.0f}")
    print('    ' + ' '.join(f"{r['fold']}={r['pnl']:+.0f}" for r in res[0]))
    for v, rs in zip(vs[1:], res[1:]):
        p = [r['pnl'] for r in rs]
        d = [x - y for x, y in zip(p, b)]
        sd = statistics.stdev(d) if len(d) > 1 else 0
        t = statistics.fmean(d) / (sd / math.sqrt(len(d))) if sd else 0
        br = sum(r['breach'] for r in rs)
        print(f"{json.dumps(v):60} d {statistics.fmean(d):+7.0f} won {sum(x > 0 for x in d):2d}/{len(d)} t {t:+5.1f}"
              f" | mean {statistics.fmean(p):+7.0f} min {min(p):+7.0f}" + (f" BREACH {br:.0f}" if br else ''), flush=True)


if __name__ == '__main__':
    main()
