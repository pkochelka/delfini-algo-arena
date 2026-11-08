"""Anchored walk-forward Optuna tuning of a round-4 strategy's PARAMS (fastsim, real round-4 tape).

    uv run python strategies/peter/r4_optuna_wf.py [STRAT.py] [--trials 45] [--jobs 9] [--prune-frac 0.03]

Steps (train -> out-of-sample test, all windows from data/splits_r4):
    A  wf01-04 -> wf05-08
    B  wf01-08 -> wf09-12
    C  wf01-12 -> hold (4 hold16 folds past 4500 s, never tuned on)
Each step is its own TPE study (trial 0 = the file's own PARAMS); the best train trial is then run on the
test windows next to the base. Report: paired test delta per step, plus whether a parameter's chosen value is
stable across steps. Studies live in logs/r4/optuna_wf.db (resumable), summary in logs/r4/optuna_wf.json.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import optuna

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
from r4_eval import run_one, tstat, windows  # noqa: E402

# name: (low, high, log)
SPACE = {
    'kv_tau_c': (0.15, 2.0, True),
    'kv_tau_v': (0.5, 6.0, True),
    'kv_sd_c': (0.05, 1.5, True),
    'kv_sd_v': (0.5, 8.0, True),
    'c_v': (0.03, 0.09, False),
    'theta': (0.10, 0.60, False),
    'take': (0.0, 0.40, False),
    'take_sd_t': (0.3, 2.5, False),
    'vf_th': (1.0, 3.5, False),
    'vf_keep': (0.8, 3.0, False),
    'pv_thr': (0.4, 0.95, False),
}


def base_params(strategy: str) -> dict:
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location('_strat', ROOT / strategy)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return {k: m.PARAMS[k] for k in SPACE}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('strategy', nargs='?', default='strategies/marek/round4/r4_best.py')
    ap.add_argument('--trials', type=int, default=45)
    ap.add_argument('--jobs', type=int, default=9)
    ap.add_argument('--prune-frac', type=float, default=0.03, help='prune when stage-1 PnL trails the base by this share')
    a = ap.parse_args()

    wf, _ = windows('wf')
    hold, _ = windows('hold')
    steps = [('A', wf[:4], wf[4:8]), ('B', wf[:8], wf[8:12]), ('C', wf, hold)]
    base = base_params(a.strategy)
    print('base', json.dumps(base), flush=True)
    storage = f"sqlite:///{(ROOT / 'logs' / 'r4' / 'optuna_wf.db').as_posix()}"
    tag = Path(a.strategy).stem
    summary = {'strategy': a.strategy, 'base': base, 'steps': {}}
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    js = ROOT / 'logs' / 'r4' / 'optuna_wf.json'
    if js.exists():
        old = json.loads(js.read_text())
        if old.get('strategy') == a.strategy and old.get('base') == base:
            summary = old  # resume: finished steps are not re-run
    TS = optuna.trial.TrialState

    with ProcessPoolExecutor(a.jobs) as ex:
        def evaluate(params_list: list[dict], ws: list[dict]) -> list[list[float]]:
            futs = [[ex.submit(run_one, a.strategy, p, w, 0.0) for w in ws] for p in params_list]
            return [[f.result()['pnl'] for f in fs] for fs in futs]

        for name, train, test in steps:
            if name in summary['steps']:
                print(f'[{name}] done (from {js.name}), skipped', flush=True)
                continue
            study = optuna.create_study(
                study_name=f'{tag}_{name}', storage=storage, direction='maximize', load_if_exists=True,
                sampler=optuna.samplers.TPESampler(seed=7, n_startup_trials=12, multivariate=True,
                                                   constant_liar=True))
            for t in study.trials:  # in flight when a previous run was killed
                if t.state == TS.RUNNING:
                    study._storage.set_trial_state_values(t._trial_id, TS.FAIL)
            done = [t for t in study.trials if t.state == TS.COMPLETE]
            if not done:  # the base first, synchronously: pruning needs its per-window PnL
                study.enqueue_trial(base)
                t = study.ask()
                [pn] = evaluate([{k: t.suggest_float(k, lo, hi, log=lg) for k, (lo, hi, lg) in SPACE.items()}], train)
                t.set_user_attr('win', pn)
                study.tell(t, sum(pn))
                done = [t]
            bw = done[0].user_attrs['win']
            # two stages when there are enough windows: every other window first; a trial that trails the base
            # there by more than prune_frac is told an extrapolated value and its other half is never run
            n = len(train)
            st1 = list(range(0, n, 2)) if n >= 8 else list(range(n))
            st2 = [i for i in range(n) if i not in st1]
            b1 = sum(bw[i] for i in st1)
            inflight: dict = {}  # future -> (trial, params, window index)
            res: dict = {}  # trial number -> {window index: pnl}
            meta: dict = {}  # trial number -> (trial, params, stage)

            def submit(t, p, idx):
                for i in idx:
                    inflight[ex.submit(run_one, a.strategy, p, train[i], 0.0)] = (t.number, i)

            def n_finished():
                return len([t for t in study.get_trials(deepcopy=False) if t.state.is_finished()])

            pruned = 0
            while True:
                # keep about 1.5 jobs per worker queued so no worker idles at a trial boundary
                while (n_finished() + len(meta) < a.trials
                       and sum(1 for _ in inflight) < a.jobs * 3 // 2):
                    t = study.ask()
                    p = {k: t.suggest_float(k, lo, hi, log=lg) for k, (lo, hi, lg) in SPACE.items()}
                    meta[t.number], res[t.number] = (t, p, 1), {}
                    submit(t, p, st1)
                if not inflight:
                    break
                fin, _ = wait(list(inflight), return_when=FIRST_COMPLETED)
                for f in fin:
                    num, i = inflight.pop(f)
                    res[num][i] = f.result()['pnl']
                    t, p, stage = meta[num]
                    if any(num == m for m, _ in inflight.values()):
                        continue
                    if stage == 1 and st2:
                        s1 = sum(res[num][j] for j in st1)
                        if s1 < b1 - a.prune_frac * abs(b1):
                            pn = [res[num].get(j, bw[j] + (s1 - b1) / len(st1)) for j in range(n)]
                            t.set_user_attr('win', pn)
                            t.set_user_attr('pruned', 1)
                            study.tell(t, sum(pn))
                            pruned += 1
                        else:
                            meta[num] = (t, p, 2)
                            submit(t, p, st2)
                            continue
                    else:
                        pn = [res[num][j] for j in range(n)]
                        t.set_user_attr('win', pn)
                        study.tell(t, sum(pn))
                    del meta[num]
                    done = [x for x in study.get_trials(deepcopy=False) if x.state == TS.COMPLETE]
                    print(f'[{name}] {len(done):3d} trials ({pruned} pruned)  base {done[0].value / 1e3:.1f}k  '
                          f'best {study.best_value / 1e3:.1f}k (#{study.best_trial.number})  '
                          f'#{num} {sum(pn) / 1e3:.1f}{"p" if t.user_attrs.get("pruned") else ""}', flush=True)
            # best = best fully evaluated trial (pruned values are extrapolations)
            done = [t for t in study.trials if t.state == TS.COMPLETE and not t.user_attrs.get('pruned')]
            done.sort(key=lambda t: t.number)
            study_best = max(done, key=lambda t: t.value)
            bt, b0 = study_best, done[0]
            best = {k: bt.params[k] for k in SPACE}
            tb, tv = evaluate([base, best], test)
            d = [x - y for x, y in zip(tv, tb)]
            dtr = [x - y for x, y in zip(bt.user_attrs['win'], b0.user_attrs['win'])]
            # trials ranked by train value: how many of the top 5 also beat the base out of sample is a cheap
            # second opinion against picking a lucky single trial
            top = sorted(done, key=lambda t: -t.value)[:5]
            topv = evaluate([{k: t.params[k] for k in SPACE} for t in top[1:]], test)
            top_d = [sum(tv) - sum(tb)] + [sum(v) - sum(tb) for v in topv]
            res = {'train_windows': len(train), 'test_windows': len(test), 'trials': len(done),
                   'train_base': b0.value, 'train_best': bt.value, 'train_delta_t': tstat(dtr),
                   'test_base': sum(tb), 'test_best': sum(tv), 'test_delta': sum(d), 'test_t': tstat(d),
                   'test_won': sum(x > 0 for x in d), 'test_win_delta': d, 'top5_test_delta': top_d, 'best': best}
            summary['steps'][name] = res
            print(f'[{name}] train {b0.value / 1e3:.1f}k -> {bt.value / 1e3:.1f}k (t {tstat(dtr):+.1f}) | '
                  f'TEST base {sum(tb) / 1e3:.1f}k best {sum(tv) / 1e3:.1f}k d {sum(d) / 1e3:+.1f}k t {tstat(d):+.1f} '
                  f'won {res["test_won"]}/{len(d)} | top5 test d {" ".join(f"{x / 1e3:+.1f}" for x in top_d)}',
                  flush=True)
            print(f'[{name}] best {json.dumps({k: round(v, 3) for k, v in best.items()})}', flush=True)
            (ROOT / 'logs' / 'r4' / 'optuna_wf.json').write_text(json.dumps(summary, indent=1))

    oos = sum(s['test_delta'] for s in summary['steps'].values())
    print(f'walk-forward OOS total delta {oos / 1e3:+.1f}k over {sum(len(s[2]) for s in steps)} windows', flush=True)
    for k in SPACE:
        print(f'  {k:10} base {base[k]!s:8} ' + ' '.join(f'{n}:{s["best"][k]:.3f}' for n, s in summary['steps'].items()))


if __name__ == '__main__':
    main()
