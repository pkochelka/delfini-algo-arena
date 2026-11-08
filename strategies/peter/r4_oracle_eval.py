"""Full round-4 tape runs of r4_oracle.py (or any strategy) variants, PnL split into calm / volatile / shock.

    uv run python strategies/peter/r4_oracle_eval.py '{}' '{"or_lvl":1}' ... [--strategy F.py] [--workers 8]
                                                     [--tape data/round4.csv]
Labels: Marek's flow/regimes_1s.csv + flow/shock_events.csv (shock = onset - 1 .. onset + 20 s), via strategy_v2/eval4.py.
Each line: total, calm, volatile, shock, delta vs the first variant, fills, breach. Cache: logs/r4/oracle_cache.json.
"""
import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fastsim.pool import Job, run_many  # noqa: E402

_s = importlib.util.spec_from_file_location('eval4', ROOT / 'marek/research/round4/strategy_v2/eval4.py')
E4 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(E4)
CACHE = ROOT / 'logs/r4/oracle_cache.json'
DEPS = [ROOT / 'strategies/marek/round4/r4_best.py']


def key(strat, v, tape):
    h = hashlib.sha1(Path(strat).read_bytes())
    for d in DEPS:
        h.update(d.read_bytes())
    return f"{h.hexdigest()[:12]}|{json.dumps(v, sort_keys=True)}|{tape}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('variants', nargs='*', default=['{}'])
    ap.add_argument('--strategy', default=str(ROOT / 'strategies/peter/r4_oracle.py'))
    ap.add_argument('--tape', default='data/round4.csv')
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    vs = [json.loads(v) for v in a.variants]
    ks = [key(a.strategy, v, a.tape) for v in vs]
    todo = [(k, v) for k, v in zip(ks, vs) if k not in cache]
    todo = list({k: v for k, v in todo}.items())
    if todo:
        outs = run_many([Job(a.strategy, a.tape, v) for _, v in todo], a.workers)
        ctx = E4.load_ctx(a.tape)
        for (k, v), o in zip(todo, outs):
            r = E4.split(o, *ctx)
            if 'err' in r:
                print('ERR', v, r['err'][-1500:])
                continue
            cache[k] = {'pnl': r['pnl'], 'parts': r['parts'], 'win': r['win'], 'fills': r.get('fills'),
                        'breach': r.get('breach_penalty', 0), 'per_shock': r['per_shock']}
        disk = json.loads(CACHE.read_text()) if CACHE.exists() else {}
        disk.update(cache)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(disk))
    b = cache.get(ks[0])
    for v, k in zip(vs, ks):
        r = cache.get(k)
        if r is None:
            continue
        P = r['parts']
        d = (r['pnl'] - b['pnl']) / 1e3 if b else 0.0
        print(f"{json.dumps(v)[:64]:64} {r['pnl'] / 1e3:8.1f}k  calm {P['calm'] / 1e3:7.1f}  vol {P['volatile'] / 1e3:7.1f}"
              f"  shock {P['shock'] / 1e3:7.1f}  d {d:+8.1f}k  fills {r['fills']}  breach {r['breach'] / 1e3:.1f}k",
              flush=True)


if __name__ == '__main__':
    main()
