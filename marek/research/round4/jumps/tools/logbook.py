"""Every 50 ms of a strategy run: t, F, sd, kv, Y, turb, pos, bb, ba + market depth per tick (20 levels each side)
-> cache/book_<strat>.npz"""
import argparse, ast, pathlib, sys
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from fastsim.engine import load_strategy, run_backtest
from fastsim.tape import ensure_cache
ap = argparse.ArgumentParser(); ap.add_argument('strat'); ap.add_argument('--tape', default='data/round4.csv')
ap.add_argument('--set', nargs='*', default=[]); ap.add_argument('--dt', type=float, default=0.05)
a = ap.parse_args()
over = {k: ast.literal_eval(v) for k, v in (kv.split('=', 1) for kv in a.set)}
s = load_strategy(a.strat, over)
rows, db, da = [], [], []; nxt = [0.0]
orig = s.on_order_book_change
def hook(snap, trades):
    orig(snap, trades)
    t = snap.timestamp_us * 1e-6
    if t >= nxt[0] and s.f is not None and snap.bids and snap.asks:
        nxt[0] = t + a.dt
        bb, ba = snap.bids[0].price, snap.asks[0].price
        b = np.zeros(20, np.int32); q = np.zeros(20, np.int32)
        for o in snap.bids:
            k = bb - o.price
            if k < 20 and o.owner == 'market': b[k] += o.quantity
        for o in snap.asks:
            k = o.price - ba
            if k < 20 and o.owner == 'market': q[k] += o.quantity
        rows.append((t, s.f.mean, s.f.sd, s.f.kv, s.f.Y, float(s.turb), s.pos, bb, ba)); db.append(b); da.append(q)
s.on_order_book_change = hook
ensure_cache(ROOT / a.tape)
res, curve = run_backtest(s, ROOT / a.tape)
print('pnl', res.pnl)
out = pathlib.Path(__file__).parents[1] / 'cache' / f"book_{pathlib.Path(a.strat).stem}.npz"
np.savez(out, x=np.array(rows), db=np.array(db), da=np.array(da), curve=np.array(curve))
print(out)
