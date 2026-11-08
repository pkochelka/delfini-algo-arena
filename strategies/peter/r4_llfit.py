"""Maximum-likelihood profile of the S-filter parameters of r4_trend (r3_dev_best's FV + ladder filter), in place.

    uv run python strategies/peter/r4_llfit.py '{"c":0.02}' '{"c":0.014}' [--folds 1,3,5,7,9,11] [--dir data/splits_r4/wf]

Runs the strategy without trading on each fold and reads the filter's accumulated one-step-ahead log-likelihood of
the market maker inserts it scored (FV.ll / FV.n_ll). The base ({} = the file's PARAMS) is always run first; each
variant prints its total LL difference (nats) and per-insert mean. Higher = the variant describes the tape better.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def run(strategy, over, path):
    sys.path.insert(0, str(ROOT))
    from fastsim.engine import load_strategy, run_backtest
    s = load_strategy(ROOT / strategy, over | {'trade_start_us': 10**15})
    run_backtest(s, ROOT / path)
    return s.f.ll, s.f.n_ll


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('variants', nargs='*')
    ap.add_argument('--strategy', default='strategies/peter/r4_trend.py')
    ap.add_argument('--dir', default='data/splits_r4/wf')
    ap.add_argument('--folds', default='1,3,5,7,9,11')
    ap.add_argument('--base', default='{}')
    ap.add_argument('--jobs', type=int, default=8)
    a = ap.parse_args()
    fs = json.loads((ROOT / a.dir / 'folds.json').read_text())['folds']
    paths = [f'{a.dir}/{fs[int(i) - 1]["file"]}' for i in a.folds.split(',')]
    base = json.loads(a.base)
    vs = [{}] + [json.loads(v) for v in a.variants]
    with ProcessPoolExecutor(a.jobs) as ex:
        futs = [[ex.submit(run, a.strategy, base | v, p) for p in paths] for v in vs]
        res = [[f.result() for f in fl] for fl in futs]
    b = sum(r[0] for r in res[0])
    n = sum(r[1] for r in res[0])
    print(f'base {json.dumps(base)}: LL {b:.0f} over {n} inserts ({b / n:.4f}/insert)')
    for v, r in zip(vs[1:], res[1:]):
        ll = sum(x[0] for x in r)
        per = [x[0] - y[0] for x, y in zip(r, res[0])]
        print(f'{json.dumps(v)[:60]:60} dLL {ll - b:+9.1f}  folds won {sum(d > 0 for d in per)}/{len(per)}  '
              f'({" ".join(f"{d:+.0f}" for d in per)})')


if __name__ == '__main__':
    main()
