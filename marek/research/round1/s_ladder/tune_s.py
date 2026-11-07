"""Optuna search over s_ladder's S parameters on round1 folds wf01-10 (total PnL). Trials -> tune_results.jsonl.

    uv run python marek/research/round1/s_ladder/tune_s.py N_TRIALS
"""
import json, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import optuna
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'strategies' / 'peter'))
from tune import load_folds, run_fold  # noqa: E402
STRAT = str(ROOT / 'strategies' / 'marek' / 's_ladder.py')
FOLDS = load_folds(ROOT / 'data/splits_round1/wf', list(range(1, 11)))
OUT = Path(__file__).with_name('tune_results.jsonl')


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    st = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=1, multivariate=True))
    st.enqueue_trial({'s_pull': 0.3, 's_pull_flat': 0.0, 's_pull_pos': 0.0, 's_take': 0.1, 's_improve': 0.4,
                      's_hyst': 0.0, 's_maxdepth': 3, 's_gap': 0.05, 's_take_max': 200, 'topup': 10})
    with ProcessPoolExecutor(max_workers=10) as pool:
        def obj(tr):
            ov = {'s_pull': tr.suggest_float('s_pull', 0.15, 0.5), 's_pull_flat': tr.suggest_float('s_pull_flat', -0.4, 0.3),
                  's_pull_pos': tr.suggest_float('s_pull_pos', -0.3, 0.1), 's_take': tr.suggest_float('s_take', 0.02, 0.3),
                  's_improve': tr.suggest_float('s_improve', 0.2, 0.7), 's_hyst': tr.suggest_float('s_hyst', 0.0, 0.1),
                  's_maxdepth': tr.suggest_int('s_maxdepth', 1, 3), 's_gap': tr.suggest_float('s_gap', 0.0, 0.2),
                  's_take_max': tr.suggest_categorical('s_take_max', [50, 100, 200]),
                  'topup': tr.suggest_categorical('topup', [1, 10, 30])}
            pn = [r['pnl'] for r in pool.map(run_fold, [STRAT] * 10, [ov] * 10, FOLDS)]
            with OUT.open('a') as f:
                f.write(json.dumps({'params': ov, 'total': sum(pn), 'folds': pn}) + '\n')
            print(f"{tr.number:3d} {sum(pn):10,.0f} {ov}", flush=True)
            return sum(pn)
        st.optimize(obj, n_trials=n)
    print('BEST', st.best_value, st.best_params)


if __name__ == '__main__':
    main()
