"""Ceiling test for 'a better S filter': run the strategy with the fair value used by _quote replaced by an oracle
from s_dynamics/s_path.parquet: `filt` (the research group's causal reference filter, error sd 0.26 vs ours 0.43 in
volatile) or `hat` (the smoothed, non-causal path: an absolute bound). The filter itself still runs (kv, pv, sd, the
shock sniper use it); only F inside _quote is swapped. `--sd` also swaps sd for s_filt_sd.
uv run --with pyarrow python marek/research/round4/edge/tools/oracle.py STRAT --mode filt|hat [--sd] [--set k=v ...]"""
import argparse, ast, pathlib, sys
import numpy as np, pandas as pd
ROOT = pathlib.Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from fastsim.engine import load_strategy, run_backtest
from fastsim.tape import ensure_cache
R4 = pathlib.Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser(); ap.add_argument('strat'); ap.add_argument('--mode', default='filt')
ap.add_argument('--sd', action='store_true'); ap.add_argument('--set', nargs='*', default=[])
ap.add_argument('--tape', default='data/round4.csv')
a = ap.parse_args()
over = {k: ast.literal_eval(v) for k, v in (kv.split('=', 1) for kv in a.set)}
sp = pd.read_parquet(R4 / 's_dynamics/s_path.parquet', columns=['t_us', 's_hat', 's_filt', 's_filt_sd'])
st = sp.t_us.values * 1e-6
SO = (sp.s_hat if a.mode == 'hat' else sp.s_filt).values
SD = sp.s_filt_sd.values
s = load_strategy(a.strat, over)
QN = '_quote' if hasattr(s, '_quote') else '_trade'
orig_q = getattr(s, QN)
stat = {'n': 0, 'e2': 0.0}


def q(f, *args):
    i = min(max(np.searchsorted(st, s.now_t, 'right') - 1, 0), len(SO) - 1)
    m0, sd0 = f.mean, f.sd
    stat['n'] += 1; stat['e2'] += (m0 - SO[i]) ** 2
    f.mean = float(SO[i])
    if a.sd:
        f.sd = float(SD[i])
    try:
        orig_q(f, *args)
    finally:
        f.mean, f.sd = m0, sd0


setattr(s, QN, q)
ensure_cache(ROOT / a.tape)
res, curve = run_backtest(s, ROOT / a.tape)
print(f'mode {a.mode} sd {a.sd} set {over}: pnl {res.pnl:.0f} breach {res.breach_penalty:.0f} '
      f'rmse(F - oracle) {np.sqrt(stat["e2"] / max(stat["n"], 1)):.3f}')
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))
from eval4 import split, load_ctx
out = {'result': {'pnl': res.pnl, 'breach_penalty': res.breach_penalty}, 'curve': [(t_, p_) for t_, p_ in curve]}
r = split(out, *load_ctx(a.tape))
print(json.dumps({k: r[k] for k in ('parts', 'win_cv')}))
with open(Path(__file__).parents[1] / 'results.jsonl', 'a') as fh:
    fh.write(json.dumps({'tag': f'oracle_{a.mode}' + ('_sd' if a.sd else ''), 'file': a.strat, 'tape': a.tape, 'set': over,
                         'pnl': round(res.pnl), 'breach': round(res.breach_penalty), 'parts': r['parts'], 'win': r['win'],
                         'win_cv': r['win_cv'], 'per_shock': r['per_shock']}) + '\n')
