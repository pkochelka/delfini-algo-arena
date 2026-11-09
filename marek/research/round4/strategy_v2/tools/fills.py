"""Log our fills with context: t_us, side (+1 buy), qty, price, kind (0 passive / 1 IOC), F, sd, pv, turb, pos_before,
bb, ba (market touch at fill), jl side (live joiner episode), queue ahead at post (lots, passive only).
uv run --with pyarrow --with numba python fills.py STRAT [--set k=v] [--out F]  -> cache/fills_<stem>.npy"""
import argparse, ast, pathlib, sys
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from fastsim.engine import load_strategy, run_backtest
from fastsim.tape import ensure_cache
from simulator.models import OrderStatus, OrderType, Side
ap = argparse.ArgumentParser(); ap.add_argument('strat'); ap.add_argument('--tape', default='data/round4.csv')
ap.add_argument('--set', nargs='*', default=[]); ap.add_argument('--out', default=None)
a = ap.parse_args()
over = {k: ast.literal_eval(v) for k, v in (kv.split('=', 1) for kv in a.set)}
s = load_strategy(a.strat, over)
kind = {}; post = {}; rows = []; snap_ = [None]
osend = s.send_order
def send(req):
    oid = osend(req)
    kind[oid] = 1 if req.order_type is OrderType.IOC else 0
    sn = snap_[0]
    q = 0
    if sn is not None and req.order_type is not OrderType.IOC:
        if req.side is Side.BUY:
            q = sum(v.quantity for v in sn.bids if v.price >= req.price)
        else:
            q = sum(v.quantity for v in sn.asks if v.price <= req.price)
    post[oid] = (q, s.now_t)
    return oid
s.send_order = send
oob = s.on_order_book_change
def ob(snap, trades):
    snap_[0] = snap
    oob(snap, trades)
s.on_order_book_change = ob
opu = s.on_private_order_update
def pu(ev):
    st = ev.status
    o = s.orders.get(ev.order_id)
    if o is not None and (st is OrderStatus.PARTIALLY_FILLED or st is OrderStatus.FILLED) and ev.filled_quantity:
        f = s.f; sn = snap_[0]
        js = s._jl_live(s.turb) if hasattr(s, '_jl_live') else 0
        q0, t0 = post.get(ev.order_id, (0, 0))
        rows.append((s.now_t * 1e6, 1 if o[0] is Side.BUY else -1, ev.filled_quantity, o[1], kind.get(ev.order_id, 0),
                     f.mean, f.sd, f.pv, float(s.turb), s.pos, sn.bids[0].price if sn and sn.bids else 0,
                     sn.asks[0].price if sn and sn.asks else 0, js, q0, s.now_t - t0))
    opu(ev)
s.on_private_order_update = pu
ensure_cache(ROOT / a.tape)
res, curve = run_backtest(s, ROOT / a.tape)
print('pnl', res.pnl, 'fills', len(rows))
out = a.out or (pathlib.Path(__file__).parents[1] / 'cache' / f"fills_{pathlib.Path(a.strat).stem}.npy")
np.save(out, np.array(rows)); print(out)
