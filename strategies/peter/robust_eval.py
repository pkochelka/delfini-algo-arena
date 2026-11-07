"""Robustness table: mean PnL per augmentation tag (over folds) for each variant.
uv run python strategies/peter/robust_eval.py strategy.py '{"a":1}' ... [--folds wf03,wf07,wf11] [--tags orig,mirror,...]
Variant may carry "_file": other strategy path."""
import argparse, json, statistics, sys
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, 'strategies/peter')
from tune import run_fold

TAGS = 'orig mirror pscale2 tscale0.25 tscale4 vscale0.3 vscale3 drift30 drift-30 noise1 takers3 makers3 flicker3 shocks3'.split()

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('strategy'); ap.add_argument('variants', nargs='*')
    ap.add_argument('--folds', default='wf03,wf07,wf11'); ap.add_argument('--tags', default=','.join(TAGS))
    ap.add_argument('--jobs', type=int, default=12)
    a = ap.parse_args(); tags = a.tags.split(','); folds = a.folds.split(',')
    vs = [json.loads(v) for v in a.variants] or [{}]
    def path(f, tag): return f'data/splits_r1/wf/{f}.csv' if tag == 'orig' else f'data/splits_r1/aug/{f}_{tag}.csv'
    with ProcessPoolExecutor(a.jobs) as ex:
        futs = {}
        for i, v in enumerate(vs):
            strat = v.pop('_file', a.strategy)
            for tag in tags:
                for f in folds:
                    futs[i, tag, f] = ex.submit(run_fold, strat, v, {'name': f, 'path': path(f, tag), 'trade_start_us': 0})
        res = {k: x.result() for k, x in futs.items()}
    print(f'{"tag":12}' + ''.join(f'{"v" + str(i):>10}' for i in range(len(vs))))
    tot = [[] for _ in vs]
    for tag in tags:
        row = []
        for i in range(len(vs)):
            m = statistics.fmean(res[i, tag, f]['pnl'] for f in folds); row.append(m); tot[i].append(m)
        br = [sum(res[i, tag, f]['breach'] for f in folds) for i in range(len(vs))]
        print(f'{tag:12}' + ''.join(f'{x:+10.0f}' for x in row) + ('  breach ' + str(br) if any(br) else ''))
    print(f'{"MEAN":12}' + ''.join(f'{statistics.fmean(t):+10.0f}' for t in tot))
    print(f'{"WORST":12}' + ''.join(f'{min(t):+10.0f}' for t in tot))
    for i, v in enumerate(vs): print(f'v{i}: {json.dumps(v)}')

if __name__ == '__main__':
    main()
