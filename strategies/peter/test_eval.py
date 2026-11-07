"""Score PARAMS variants on round1's held-out test (9000-12000 s): whole file, trading from 9000 s.

uv run python strategies/peter/test_eval.py [--strategy path] '{"a": 1}' ...
"""
import json, sys
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, 'strategies/peter')
from tune import run_fold
if __name__ == '__main__':
    args = sys.argv[1:]
    strat = 'strategies/peter/queue_mm.py'
    if args[:1] == ['--strategy']:
        strat, args = args[1], args[2:]
    vs = [json.loads(v) for v in args] or [{}]
    f = {'name': 'TEST', 'path': 'data/round1.csv', 'trade_start_us': 9000001434}
    with ProcessPoolExecutor(4) as ex:
        for v, r in zip(vs, ex.map(run_fold, [strat] * len(vs), vs, [f] * len(vs))):
            print(json.dumps(v), r)
