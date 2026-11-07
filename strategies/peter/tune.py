"""Optuna hyperparameter search for a PARAMS-driven strategy (arb_mm by default), scored on walk-forward folds.

Folds come from `split_data.py --folds` (each file has a learn-only prefix; `trade_start_us` from
folds.json is passed as a PARAMS override, so PnL counts only the eval window).

Modes
    search        one study over --fit folds; the best params are then scored on --check folds
                  (folds the search never saw), next to the defaults.
    walkforward   validates the tuning *procedure*: for each fold k >= --min-fit, tune on the folds
                  before k (expanding, or the last --window of them) and score the winner on fold k.
                  If tuned params don't beat the defaults out of sample here, don't trust `search` either.
    eval          score the defaults or a saved --params JSON on folds (no search).

Usage
    uv run python strategies/peter/tune.py search --folds-dir data/splits_v2/wf --fit 1-8 --check 9-12 --trials 200
    uv run python strategies/peter/tune.py walkforward --folds-dir data/splits_v2/wf --min-fit 4 --trials 40
    uv run python strategies/peter/tune.py eval --folds-dir data/splits_v2/wf --fit 1-12 --params strategies/peter/tuned/x.json

Objective per trial: mean(fold PnL) - risk_k * std(fold PnL) - inv_pen * mean(abs_inventory_q95).
Studies are stored in logs/optuna.db (resumable: rerun with the same --study). The best params are
written to strategies/peter/tuned/<study>.json, which is meant to be committed.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

# name -> (kind, low, high, log) or ('cat', [choices]); kind is 'float' or 'int'. Edit freely; anything
# not listed keeps its PARAMS default. One space per strategy file stem; SPACE is the one in use.
SPACES: dict[str, dict[str, tuple]] = {}
SPACES['ladder_mm'] = {
    'size_top': ('int', 50, 200, False),
    'topup': ('int', 0, 60, False),
    'levels': ('cat', [1, 2]),
    'size_deep': ('int', 0, 100, False),
    'tight_offset': ('cat', [0, 1]),
    'skew_lots': ('cat', [0, 25, 50, 100]),
}
SPACE: dict[str, tuple] = {
    'inv_risk': ('float', 4.0, 128.0, True),
    'size_frac': ('float', 0.05, 0.6, False),
    'tau': ('float', 10.0, 300.0, True),
    'adapt': ('float', 0.0, 6.0, False),
    'delta_mult': ('float', 0.2, 1.5, False),
    'take_margin': ('float', 0.0, 3.0, False),
    'take_frac': ('float', 0.05, 0.5, False),
    'post_k': ('float', 0.25, 4.0, False),
    'post_target': ('float', 0.01, 0.15, True),
    'pred_weight': ('float', 0.0, 2.0, False),
    'takes': ('cat', [0, 1]),
}
SPACES['arb_mm'] = dict(SPACE)


def parse_range(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(','):
        a, _, b = part.partition('-')
        out.extend(range(int(a), int(b or a) + 1))
    return out


def load_folds(folds_dir: Path, idx: list[int]) -> list[dict]:
    meta = json.loads((folds_dir / 'folds.json').read_text())['folds']
    by_num = {int(m['name'][2:]): m | {'path': str(folds_dir / m['file'])} for m in meta}
    return [by_num[i] for i in idx]


def run_fold(strategy: str, overrides: dict, fold: dict) -> dict:
    """Worker: replay one fold with PARAMS overrides, return the simulator's result fields."""
    sys.path.insert(0, str(HERE))
    from arena import load_agent  # private module copy per call, so overrides never leak
    from simulator.main import (MAX_POSITION, MAX_POTENTIAL_POSITION, ORDER_ACCEPT_DELAY_MS,
                                POSITION_BREACH_PENALTY_PER_SECOND)
    from simulator.simulator import ExchangeSimulator, SimulationConfig

    strat = load_agent(Path(strategy), overrides | {'trade_start_us': fold['trade_start_us']}, 'tune')
    cfg = SimulationConfig(max_position=MAX_POSITION, max_potential_position=MAX_POTENTIAL_POSITION,
                           position_breach_penalty_per_second=POSITION_BREACH_PENALTY_PER_SECOND,
                           order_accept_delay_ms=ORDER_ACCEPT_DELAY_MS)
    r = ExchangeSimulator(strategy=strat, config=cfg).run(Path(fold['path']))
    return {'fold': fold['name'], 'pnl': r.pnl, 'fills': r.fills, 'q95': r.abs_inventory_q95,
            'breach': r.breach_penalty}


class Runner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.pool = ProcessPoolExecutor(max_workers=args.jobs)

    def evaluate(self, overrides: dict, folds: list[dict]) -> list[dict]:
        futs = [self.pool.submit(run_fold, self.args.strategy, overrides, f) for f in folds]
        return [f.result() for f in futs]

    def score(self, res: list[dict]) -> float:
        pnls = [r['pnl'] for r in res]
        sd = statistics.pstdev(pnls) if len(pnls) > 1 else 0.0
        inv = statistics.fmean(r['q95'] for r in res)
        return statistics.fmean(pnls) - self.args.risk_k * sd - self.args.inv_pen * inv


def suggest(trial) -> dict:
    out = {}
    for name, spec in SPACE.items():
        if spec[0] == 'cat':
            out[name] = trial.suggest_categorical(name, spec[1])
        elif spec[0] == 'int':
            _, lo, hi, log = spec
            out[name] = trial.suggest_int(name, lo, hi, log=log)
        else:
            _, lo, hi, log = spec
            out[name] = trial.suggest_float(name, lo, hi, log=log)
    if 'post_k' in out:  # the adaptive buffer clamps post_k into [post_k_min, post_k_max]
        out['post_k_max'] = max(3.0, out['post_k'])
        out['post_k_min'] = min(0.25, out['post_k'])
    return out


def summary(label: str, res: list[dict], score: float) -> str:
    pnls = [r['pnl'] for r in res]
    pos = sum(p > 0 for p in pnls)
    cells = ' '.join(f"{r['fold']}={r['pnl']:+.0f}" for r in res)
    return (f'{label:<9} score={score:+9.1f}  mean={statistics.fmean(pnls):+8.1f}  min={min(pnls):+8.1f}  '
            f'win={pos}/{len(pnls)}  q95={statistics.fmean(r["q95"] for r in res):5.1f}\n          {cells}')


def run_study(runner: Runner, name: str, folds: list[dict], trials: int, seed: int):
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    storage = f'sqlite:///{(ROOT / "logs" / "optuna.db").as_posix()}'
    study = optuna.create_study(study_name=name, storage=storage, direction='maximize', load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=seed, multivariate=True))
    done = len(study.trials)
    if not done:
        study.enqueue_trial({k: v for k, v in DEFAULTS.items() if k in SPACE})  # baseline as trial 0
    t_start = time.time()

    def objective(trial):
        params = suggest(trial)
        res = runner.evaluate(params, folds)
        for r in res:
            trial.set_user_attr(r['fold'], r['pnl'])
        return runner.score(res)

    def log(study, trial):
        if trial.value is not None:
            print(f'  [{name}] trial {trial.number:4d}  {trial.value:+9.1f}  best {study.best_value:+9.1f}'
                  f'  ({time.time() - t_start:6.0f}s)', flush=True)

    remaining = max(0, trials - done)
    study.optimize(objective, n_trials=remaining, callbacks=[log])
    return study


def best_params(study) -> dict:
    p = dict(study.best_params)
    if 'post_k' in p:
        p['post_k_max'] = max(3.0, p['post_k'])
        p['post_k_min'] = min(0.25, p['post_k'])
    return p


DEFAULTS: dict = {}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('mode', choices=['search', 'walkforward', 'eval'])
    ap.add_argument('--strategy', default=str(HERE / 'arb_mm.py'))
    ap.add_argument('--folds-dir', type=Path, default=ROOT / 'data/splits_v2/wf')
    ap.add_argument('--fit', default='1-8', help='fold numbers to tune on, e.g. 1-8 or 1,3,5-7')
    ap.add_argument('--check', default='9-12', help='search: out-of-sample folds scored after tuning')
    ap.add_argument('--min-fit', type=int, default=4, help='walkforward: first fold scored out of sample')
    ap.add_argument('--window', type=int, default=0, help='walkforward: tune on only the last N folds (0 = expanding)')
    ap.add_argument('--trials', type=int, default=150)
    ap.add_argument('--study', default=None, help='study name (default derived from mode and folds)')
    ap.add_argument('--params', type=Path, help='eval: JSON of overrides')
    ap.add_argument('--risk-k', type=float, default=0.5, help='penalty per std of fold PnL')
    ap.add_argument('--inv-pen', type=float, default=0.0, help='penalty per unit of mean abs_inventory_q95')
    ap.add_argument('--jobs', type=int, default=None, help='parallel fold replays (default: CPU count)')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    sys.path.insert(0, str(HERE))
    from arena import load_agent
    DEFAULTS.update(load_agent(Path(args.strategy), {}, 'defaults').p)

    runner = Runner(args)
    out_dir = HERE / 'tuned'
    out_dir.mkdir(exist_ok=True)
    tag = Path(args.strategy).stem
    if tag in SPACES:
        SPACE.clear()
        SPACE.update(SPACES[tag])

    if args.mode == 'eval':
        folds = load_folds(args.folds_dir, parse_range(args.fit))
        over = json.loads(args.params.read_text())['params'] if args.params else {}
        res = runner.evaluate(over, folds)
        print(summary('params' if over else 'defaults', res, runner.score(res)))
        return

    if args.mode == 'search':
        fit = load_folds(args.folds_dir, parse_range(args.fit))
        check = load_folds(args.folds_dir, parse_range(args.check)) if args.check else []
        name = args.study or f'{tag}_search_f{args.fit}'
        study = run_study(runner, name, fit, args.trials, args.seed)
        best = best_params(study)
        print(f'\nbest of {len(study.trials)} trials on fit folds {args.fit}: {study.best_value:+.1f}')
        for k, v in best.items():
            print(f'  {k:<12} {DEFAULTS.get(k)!s:>8} -> {v:.4g}' if isinstance(v, float) else f'  {k:<12} -> {v}')
        report = {'study': name, 'strategy': tag, 'fit': args.fit, 'check': args.check,
                  'objective': {'risk_k': args.risk_k, 'inv_pen': args.inv_pen},
                  'fit_score': study.best_value, 'params': best}
        if check:
            d, b = runner.evaluate({}, check), runner.evaluate(best, check)
            print(f'\nout of sample, folds {args.check}:')
            print(summary('defaults', d, runner.score(d)))
            print(summary('tuned', b, runner.score(b)))
            report['check_defaults'] = d
            report['check_tuned'] = b
        try:
            import optuna
            imp = optuna.importance.get_param_importances(study)
            print('\nparam importance (fANOVA): ' + ', '.join(f'{k}={v:.2f}' for k, v in imp.items()))
            report['importance'] = imp
        except Exception as e:  # needs enough finished trials
            print(f'(importance unavailable: {e})')
        path = out_dir / f'{name}.json'
        path.write_text(json.dumps(report, indent=2))
        print(f'\nwrote {path}')
        return

    # walkforward
    meta = json.loads((args.folds_dir / 'folds.json').read_text())['folds']
    n = len(meta)
    rows = []
    for k in range(args.min_fit, n + 1):
        lo = 1 if not args.window else max(1, k - args.window)
        fit = load_folds(args.folds_dir, list(range(lo, k)))
        test = load_folds(args.folds_dir, [k])
        name = (args.study or f'{tag}_wf') + f'_fit{lo}-{k - 1}'
        study = run_study(runner, name, fit, args.trials, args.seed)
        best = best_params(study)
        d, b = runner.evaluate({}, test)[0], runner.evaluate(best, test)[0]
        rows.append({'fold': k, 'fit': f'{lo}-{k - 1}', 'fit_score': study.best_value,
                     'default_pnl': d['pnl'], 'tuned_pnl': b['pnl'], 'params': best})
        print(f'fold {k:2d}: tuned on {lo}-{k - 1}  default {d["pnl"]:+8.1f}  tuned {b["pnl"]:+8.1f}', flush=True)
    dd = [r['default_pnl'] for r in rows]
    tt = [r['tuned_pnl'] for r in rows]
    wins = sum(t > d for t, d in zip(tt, dd))
    print(f'\nwalk-forward OOS over folds {args.min_fit}-{n}: default mean {statistics.fmean(dd):+.1f}, '
          f'tuned mean {statistics.fmean(tt):+.1f}, tuned beats default on {wins}/{len(rows)}')
    for key in SPACE:  # how stable is each pick across steps?
        vals = [r['params'][key] for r in rows if key in r['params']]
        if vals and all(isinstance(v, (int, float)) for v in vals):
            m = statistics.fmean(vals)
            cv = statistics.pstdev(vals) / abs(m) if m else math.inf
            print(f'  {key:<12} picks {", ".join(f"{v:.3g}" for v in vals)}   (cv {cv:.2f})')
    path = out_dir / f'{(args.study or tag + "_wf")}_walkforward.json'
    path.write_text(json.dumps({'objective': {'risk_k': args.risk_k, 'inv_pen': args.inv_pen},
                                'window': args.window, 'trials': args.trials, 'steps': rows}, indent=2))
    print(f'wrote {path}')


if __name__ == '__main__':
    main()
