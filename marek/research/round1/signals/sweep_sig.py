"""A/B sig_ladder variants on the round1 walk-forward folds wf01-10 (3 worker processes).

    uv run python research/round1/signals/sweep_sig.py NAME [NAME ...]     # names from VARIANTS
    uv run python research/round1/signals/sweep_sig.py --json '{"g_share":0.1}' label

Prints total PnL, per-fold delta vs the ladder baseline (mean and sd of the paired delta).
Results are appended to research/round1/signals/results.jsonl.
"""
from __future__ import annotations

import json
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'strategies' / 'peter'))
from tune import load_folds, run_fold  # noqa: E402

STRAT = str(ROOT / 'strategies' / 'marek' / 'sig_ladder.py')
FOLDS = load_folds(ROOT / 'data/splits_round1/wf', list(range(1, 11)))
OUT = Path(__file__).with_name('results.jsonl')

# sig_ladder's PARAMS now default to the winner; every variant below is relative to the overlays OFF
# (= ladder_mm), which is what they were run against.
OFF = {'g_share': 0.0, 'g_hyst': 0.05, 'g_dep': 0.0, 'flow_k': 0.0, 'g_keep_flat': 0, 'pre_join': 0}

VARIANTS: dict[str, dict] = {
    'base': {},
    'shrink s.10': {'g_share': 0.10},
    'shrink s.05': {'g_share': 0.05},
    'shrink s.15': {'g_share': 0.15},
    'cancel s.10': {'g_share': 0.10, 'g_mode': 'cancel'},
    'freeze s.10': {'g_share': 0.10, 'g_mode': 'freeze'},
    'keepflat s.10': {'g_share': 0.10, 'g_keep_flat': 1},
    'shrink q50': {'g_qty': 50},
    'prejoin s.10': {'g_share': 0.10, 'pre_join': 1},
    # level depletion: only an old level being eaten counts, not a fresh one being built
    's.10 dep.5': {'g_share': 0.10, 'g_dep': 0.5},
    's.20 dep.5': {'g_share': 0.20, 'g_dep': 0.5},
    's.10 dep.3': {'g_share': 0.10, 'g_dep': 0.3},
    's.05 dep.3': {'g_share': 0.05, 'g_dep': 0.3},
    'keepflat s.10 dep.5': {'g_share': 0.10, 'g_dep': 0.5, 'g_keep_flat': 1},
    'prejoin s.10 dep.5': {'g_share': 0.10, 'g_dep': 0.5, 'pre_join': 1},
    'freeze s.10 dep.5': {'g_share': 0.10, 'g_dep': 0.5, 'g_mode': 'freeze'},
    # narrower guards around s.05 dep.3
    's.03 dep.3': {'g_share': 0.03, 'g_dep': 0.3},
    's.05 dep.2': {'g_share': 0.05, 'g_dep': 0.2},
    's.05 dep.4': {'g_share': 0.05, 'g_dep': 0.4},
    's.07 dep.3': {'g_share': 0.07, 'g_dep': 0.3},
    's.05 dep.3 h.02': {'g_share': 0.05, 'g_dep': 0.3, 'g_hyst': 0.02},
    's.05 dep.3 h.15': {'g_share': 0.05, 'g_dep': 0.3, 'g_hyst': 0.15},
    's.05 dep.3 keep50': {'g_share': 0.05, 'g_dep': 0.3, 'g_keep': 50},
    's.05 dep.3 flow.05': {'g_share': 0.05, 'g_dep': 0.3, 'flow_k': 0.05},
    'q30 dep.3': {'g_qty': 30, 'g_dep': 0.3},
    # keep the flattening part (keeps its queue priority), drop only the part that would flip us
    'kf s.05 dep.3': {'g_share': 0.05, 'g_dep': 0.3, 'g_keep_flat': 1},
    'kf s.10 dep.3': {'g_share': 0.10, 'g_dep': 0.3, 'g_keep_flat': 1},
    'kf s.15 dep.5': {'g_share': 0.15, 'g_dep': 0.5, 'g_keep_flat': 1},
    'kf s.20 dep.5': {'g_share': 0.20, 'g_dep': 0.5, 'g_keep_flat': 1},
    'kf s.10 dep.7': {'g_share': 0.10, 'g_dep': 0.7, 'g_keep_flat': 1},
    'kfpj s.10 dep.5': {'g_share': 0.10, 'g_dep': 0.5, 'g_keep_flat': 1, 'pre_join': 1},
    'kfpj s.05 dep.3': {'g_share': 0.05, 'g_dep': 0.3, 'g_keep_flat': 1, 'pre_join': 1},
    'kfpj s.15 dep.5': {'g_share': 0.15, 'g_dep': 0.5, 'g_keep_flat': 1, 'pre_join': 1},
    'kfpj hold s.10 dep.5': {'g_share': 0.10, 'g_dep': 0.5, 'g_keep_flat': 1, 'pre_join': 1, 'pj_hold': 1},
    'pj s.05 dep.3': {'g_share': 0.05, 'g_dep': 0.3, 'pre_join': 1},
    'pj s.15 dep.5': {'g_share': 0.15, 'g_dep': 0.5, 'pre_join': 1},
    # around the winner kfpj s.10 dep.5
    'kfpj always (control)': {'g_share': 1.01, 'g_hyst': 0.0, 'g_keep_flat': 1, 'pre_join': 1},
    'kfpj s.10 nodep': {'g_share': 0.10, 'g_keep_flat': 1, 'pre_join': 1},
    'kfpj s.10 dep.7': {'g_share': 0.10, 'g_dep': 0.7, 'g_keep_flat': 1, 'pre_join': 1},
    'kfpj s.20 dep.5': {'g_share': 0.20, 'g_dep': 0.5, 'g_keep_flat': 1, 'pre_join': 1},
    'kfpj s.20 dep.7': {'g_share': 0.20, 'g_dep': 0.7, 'g_keep_flat': 1, 'pre_join': 1},
    'kfpj s.10 dep.5 h.10': {'g_share': 0.10, 'g_dep': 0.5, 'g_keep_flat': 1, 'pre_join': 1, 'g_hyst': 0.10},
    'kfpj s.10 dep.5 h.0': {'g_share': 0.10, 'g_dep': 0.5, 'g_keep_flat': 1, 'pre_join': 1, 'g_hyst': 0.0},
    'kfpj s.10 dep.5 flow.05': {'g_share': 0.10, 'g_dep': 0.5, 'g_keep_flat': 1, 'pre_join': 1, 'flow_k': 0.05},
    'kfpj s.10 dep.5 anyspread': {'g_share': 0.10, 'g_dep': 0.5, 'g_keep_flat': 1, 'pre_join': 1, 'g_spread1': 0},
    'kfpj s.10 dep.5 h.0 flow.05': {'g_share': 0.10, 'g_dep': 0.5, 'g_keep_flat': 1, 'pre_join': 1, 'g_hyst': 0.0, 'flow_k': 0.05},
    'kfpj s.10 dep.5 flow.10': {'g_share': 0.10, 'g_dep': 0.5, 'g_keep_flat': 1, 'pre_join': 1, 'flow_k': 0.10},
    'kfpj s.10 dep.5 flow.05 300ms': {'g_share': 0.10, 'g_dep': 0.5, 'g_keep_flat': 1, 'pre_join': 1, 'flow_k': 0.05, 'flow_ms': 300},
    'kfpj s.10 dep.5 h.0 flow.10': {'g_share': 0.10, 'g_dep': 0.5, 'g_keep_flat': 1, 'pre_join': 1, 'g_hyst': 0.0, 'flow_k': 0.10},
    'kfpj s.10 dep.5 flow.20': {'g_share': 0.10, 'g_dep': 0.5, 'g_keep_flat': 1, 'pre_join': 1, 'flow_k': 0.20},
    'kfpj s.10 dep.5 flow.40': {'g_share': 0.10, 'g_dep': 0.5, 'g_keep_flat': 1, 'pre_join': 1, 'flow_k': 0.40},
    'kf s.10 dep.5 flow.05': {'g_share': 0.10, 'g_dep': 0.5, 'g_keep_flat': 1, 'flow_k': 0.05},
}


def baseline() -> dict:
    for line in OUT.read_text().splitlines() if OUT.exists() else []:
        r = json.loads(line)
        if r['label'] == 'base':
            return r
    return {}


def main() -> None:
    args = sys.argv[1:]
    if args and args[0] == '--json':
        sel = {args[2]: json.loads(args[1])}
    else:
        sel = {n: VARIANTS[n] for n in (args or VARIANTS)}
    base = baseline()
    with ProcessPoolExecutor(max_workers=3) as pool:
        jobs = {name: [pool.submit(run_fold, STRAT, OFF | ov, f) for f in FOLDS] for name, ov in sel.items()}
        for name, futs in jobs.items():
            res = [f.result() for f in futs]
            pnls = [r['pnl'] for r in res]
            rec = {'label': name, 'params': sel[name], 'total': sum(pnls), 'pnls': pnls,
                   'fills': sum(r['fills'] for r in res), 'breach': sum(r['breach'] for r in res)}
            line = f"{name:<24} total={sum(pnls):>9.0f} fills={rec['fills']:>6} breach={rec['breach']:.0f}"
            if base and name != 'base':
                d = [a - b for a, b in zip(pnls, base['pnls'])]
                line += f"  delta={sum(d):>+8.0f} (per fold {statistics.fmean(d):+6.0f} sd {statistics.stdev(d):5.0f}, wins {sum(x > 0 for x in d)}/10)"
            print(line, flush=True)
            with OUT.open('a') as fh:
                fh.write(json.dumps(rec) + '\n')


if __name__ == '__main__':
    main()
