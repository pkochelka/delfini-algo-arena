"""Which market inserts reach the S filter's maker likelihood (FV.maker), vs makers' class labels.
uv run --with pyarrow --with numba python ladchk.py STRAT [--set k=v] -> cache/fed_<tag>.npy (t_us of fed inserts)"""
import argparse, ast, pathlib, sys
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from fastsim.engine import load_strategy, run_backtest
from fastsim.tape import ensure_cache
ap = argparse.ArgumentParser(); ap.add_argument('strat'); ap.add_argument('--tape', default='data/round4.csv')
ap.add_argument('--set', nargs='*', default=[]); ap.add_argument('--tag', default='x')
a = ap.parse_args()
over = {k: ast.literal_eval(v) for k, v in (kv.split('=', 1) for kv in a.set)}
s = load_strategy(a.strat, over)
mod = sys.modules[type(s).__module__]
fed = []
om = mod.FV.maker
def mk(self, t, is_bid, price, opp, count_ll=True):
    fed.append((round(t * 1e6), 1 if is_bid else -1, price, self.pv))
    return om(self, t, is_bid, price, opp, count_ll)
mod.FV.maker = mk
ensure_cache(ROOT / a.tape)
res, curve = run_backtest(s, ROOT / a.tape)
print('pnl', res.pnl, 'fed', len(fed))
np.save(pathlib.Path(__file__).parents[1] / 'cache' / f'fed_{a.tag}.npy', np.array(fed))
