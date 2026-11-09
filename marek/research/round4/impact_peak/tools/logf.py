"""Log r4 filter state every 10 ms: t, mean, sd, pv, kv, Y, turb, pos, bb, ba -> impact_peak/cache/<stem>_<tape>.npz
uv run --with pyarrow python marek/research/round4/impact_peak/tools/logf.py STRAT [--tape data/round4.csv] [--set k=v]"""
import argparse, ast, pathlib, sys
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from fastsim.engine import load_strategy, run_backtest
from fastsim.tape import ensure_cache
ap = argparse.ArgumentParser(); ap.add_argument('strat'); ap.add_argument('--tape', default='data/round4.csv')
ap.add_argument('--set', nargs='*', default=[]); ap.add_argument('--out', default=None)
a = ap.parse_args()
over = {k: ast.literal_eval(v) for k, v in (kv.split('=', 1) for kv in a.set)}
s = load_strategy(a.strat, over)
rows = []; nxt = [0.0]
orig = s.on_order_book_change
def hook(snap, trades):
    orig(snap, trades)
    t = snap.timestamp_us * 1e-6
    if t >= nxt[0] and s.f is not None and snap.bids and snap.asks:
        nxt[0] = t + 0.01
        f = s.f
        rows.append((t, f.mean, f.sd, f.pv, f.kv, f.Y, float(s.turb), s.pos, snap.bids[0].price, snap.asks[0].price,
                     float(s.sg_t is not None)))
s.on_order_book_change = hook
ensure_cache(ROOT / a.tape)
res, curve = run_backtest(s, ROOT / a.tape)
print('pnl', res.pnl)
d = pathlib.Path(__file__).parents[1] / 'cache'; d.mkdir(exist_ok=True)
out = a.out or (d / f"{pathlib.Path(a.strat).stem}_{pathlib.Path(a.tape).stem}.npz")
np.savez(out, x=np.array(rows), curve=np.array(curve))
print(out)
