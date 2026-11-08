"""Run a strategy once and print its internal counters (shocks detected, snipes, TWAP episodes, joiner drops) + drought times."""
import sys, pathlib, ast
ROOT = pathlib.Path(__file__).resolve().parents[5]; sys.path.insert(0, str(ROOT))
from fastsim.engine import load_strategy, run_backtest
over = {k: ast.literal_eval(v) for k, v in (kv.split('=', 1) for kv in sys.argv[3:])}
s = load_strategy(sys.argv[1], over)
sg = []
orig = s._shock
def hook(f):
    before = s.sg_t
    r = orig(f)
    if s.sg_t is not None and s.sg_t != before: sg.append(round(s.sg_t - s.t0, 2))
    return r
s._shock = hook
res, _ = run_backtest(s, ROOT / sys.argv[2])
print('pnl', round(res.pnl), {k: getattr(s, k, None) for k in ('n_shocks', 'n_snipes', 'n_tw', 'n_j15', 'n_lad', 'jumps')})
print('drought detections', len(sg), sg)
