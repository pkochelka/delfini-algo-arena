"""Evaluate strategies / PARAMS variants on the regime split (regime_split.py), per regime and set.

    uv run python strategies/peter/regime_eval.py strategies/marek/robust_ladder.py '{}' '{"kill":0}' \\
        [--dir data/splits_r2/regime] [--regime calm,turb] [--set dev] [--jobs 12] [--detail] [--out res.jsonl]

Each stretch is a fresh replay that trades from its eval_start (trade_start_us override) and is marked to mid
at its end. Several strategies: pass `path.py::{json}` items instead of bare JSON variants.
"""
import argparse, json, statistics, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tune import run_fold  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('strategy'); ap.add_argument('variants', nargs='*')
    ap.add_argument('--dir', default='data/splits_r2/regime'); ap.add_argument('--regime', default='calm,turb')
    ap.add_argument('--set', default='dev,test'); ap.add_argument('--jobs', type=int, default=12)
    ap.add_argument('--detail', action='store_true'); ap.add_argument('--out', default='')
    a = ap.parse_args()
    meta = json.loads((Path(a.dir) / 'regime.json').read_text())['stretches']
    meta = [m for m in meta if m['regime'] in a.regime.split(',') and m['set'] in a.set.split(',')]
    folds = [{'name': m['name'], 'path': str(Path(a.dir) / m['file']), 'trade_start_us': m['trade_start_us'],
              'regime': m['regime'], 'set': m['set'], 'len': m['end_s'] - m['eval_start_s']} for m in meta]
    specs = []
    for v in a.variants or ['{}']:
        if '::' in v:
            s, j = v.split('::', 1); specs.append((s, json.loads(j or '{}')))
        else:
            specs.append((a.strategy, json.loads(v)))
    with ProcessPoolExecutor(a.jobs) as ex:
        futs = [[ex.submit(run_fold, s, v, f) for f in folds] for s, v in specs]
        for (s, v), fs in zip(specs, futs):
            res = [dict(x.result(), **{k: f[k] for k in ('regime', 'set', 'len')}) for x, f in zip(fs, folds)]
            label = f'{Path(s).stem} {json.dumps(v)}'
            parts = []
            for reg in a.regime.split(','):
                for st in a.set.split(','):
                    r = [x for x in res if x['regime'] == reg and x['set'] == st]
                    if not r:
                        continue
                    p = [x['pnl'] for x in r]; hrs = sum(x['len'] for x in r) / 3600
                    parts.append(f"{reg}/{st} {sum(p):+8.0f} ({sum(p)/hrs:+7.0f}/h, win {sum(x > 0 for x in p)}/{len(p)}, worst {min(p):+6.0f})")
            br = sum(x['breach'] for x in res)
            print(f'{label[:90]:90} | ' + ' | '.join(parts) + (f' | BREACH {br:.0f}' if br else ''), flush=True)
            if a.detail:
                print('    ' + ' '.join(f"{x['fold']}={x['pnl']:+.0f}" for x in res))
            if a.out:
                with open(a.out, 'a') as f:
                    f.write(json.dumps({'strategy': s, 'params': v, 'res': res}) + '\n')


if __name__ == '__main__':
    main()
