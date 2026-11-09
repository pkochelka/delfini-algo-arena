"""Log the online 15-lot joiner detections (r4 j15 detector) of a strategy run, plus the 10 ms state log.
uv run --with pyarrow --with numba python marek/research/round4/strategy_v2/tools/j15log.py STRAT [--set k=v]
-> cache/j15_<stem>.npz: det (t_us, side +1 bid / -1 ask, qty, price, prev_kind), x (t, F, sd, pv, turb, pos, bb, ba, kv), curve."""
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
over.setdefault('j15_on', 1)
s = load_strategy(a.strat, over)
rows = []; det = []; nxt = [0.0]
orig = s.on_order_book_change
def hook(snap, trades):
    orig(snap, trades)
    t = snap.timestamp_us * 1e-6
    if t >= nxt[0] and s.f is not None and snap.bids and snap.asks:
        nxt[0] = t + 0.01
        rows.append((t, s.f.mean, s.f.sd, s.f.pv, float(s.turb), s.pos, snap.bids[0].price, snap.asks[0].price, s.f.kv))
s.on_order_book_change = hook
oj = s._j15
def jhook(t_us, newb, newa, pbb, pba):
    r = oj(t_us, newb, newa, pbb, pba)
    if r:
        v = newb if newb is not None else newa
        det.append((t_us, 1 if newb is not None else -1, v.quantity, v.price, s.prev_kind))
    return r
s._j15 = jhook
ensure_cache(ROOT / a.tape)
res, curve = run_backtest(s, ROOT / a.tape)
print('pnl', res.pnl, 'det', len(det))
out = a.out or (pathlib.Path(__file__).parents[1] / 'cache' / f"j15_{pathlib.Path(a.strat).stem}.npz")
np.savez(out, x=np.array(rows), curve=np.array(curve), det=np.array(det))
print(out)
