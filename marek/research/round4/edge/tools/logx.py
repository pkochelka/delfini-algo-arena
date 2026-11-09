"""Rich run log for the edge pass -> cache/<strat>_<tape>.npz.
  x      10 ms state: t, F, sd, pv, turb, pos, bb, ba, Y, kv, sg(drought window 0/1)
  tk     every market taker event: t, sgn(+1 buy), filled qty, n levels, worst price
  od     every order we send: t, side(+1/-1), price, qty, ioc(1/0), pos, rest_same_resting, rest_same_cancelling,
         turb, in_shock_window, ss_dir(sniper active), stale qty available at or better than the IOC limit (-1 for limits), F, sd
  fl     every fill we get: t, side, price, qty, ioc, pos_before, F
  curve  PnL curve (t, pnl)
uv run --with pyarrow python marek/research/round4/edge/tools/logx.py STRAT [--tape data/round4.csv] [--set k=v ...]"""
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
X, TK, OD, FL = [], [], [], []
nxt = [0.0]
snap = [None]
orig_obc = s.on_order_book_change
orig_send = s.send_order
orig_priv = s.on_private_order_update


def obc(sn, trades):
    snap[0] = sn
    t = sn.timestamp_us * 1e-6
    q = 0; sgn = 0; lv = set(); worst = None
    for tr in trades:
        if tr.taker_owner == 'market':
            q += tr.quantity; sgn = 1 if tr.book_side == 'ask' else -1; lv.add(tr.price)
            worst = tr.price if worst is None else (max(worst, tr.price) if sgn > 0 else min(worst, tr.price))
    if q:
        TK.append((t, sgn, q, len(lv), worst))
    orig_obc(sn, trades)
    if t >= nxt[0] and s.f is not None and sn.bids and sn.asks:
        nxt[0] = t + 0.01
        X.append((t, s.f.mean, s.f.sd, s.f.pv, float(s.turb), s.pos, sn.bids[0].price, sn.asks[0].price, s.f.Y, s.f.kv,
                  float(s.sg_t is not None)))


def send(req):
    sn = snap[0]
    side = 1 if req.side is Side.BUY else -1
    ioc = req.order_type is OrderType.IOC
    rr = rc = 0
    for oid, o in s.orders.items():
        if (o[0] is Side.BUY) == (side > 0):
            if oid in s.cancelling:
                rc += o[2]
            else:
                rr += o[2]
    st = -1
    if ioc and sn is not None:
        st = 0
        for v in (sn.asks if side > 0 else sn.bids):
            if (v.price > req.price) if side > 0 else (v.price < req.price):
                break
            if v.owner == 'market':
                st += v.quantity
    OD.append((s.now_t, side, req.price, req.quantity, float(ioc), s.pos, rr, rc, float(s.turb),
               float(s.sg_t is not None), float(getattr(s, 'ss_dir', 0)), st, s.f.mean if s.f is not None else 0.0,
               s.f.sd if s.f is not None else 0.0))
    return orig_send(req)


def priv(ev):
    o = s.orders.get(ev.order_id)
    if o is not None and ev.filled_quantity and ev.fill_price is not None:
        FL.append((s.now_t, 1 if o[0] is Side.BUY else -1, ev.fill_price, ev.filled_quantity,
                   float(ev.order_id in s.cancelling), s.pos, s.f.mean if s.f is not None else 0.0))
    orig_priv(ev)


s.on_order_book_change = obc
s.send_order = send
s.on_private_order_update = priv
ensure_cache(ROOT / a.tape)
res, curve = run_backtest(s, ROOT / a.tape)
print('pnl', res.pnl, 'breach', res.breach_penalty)
out = a.out or (pathlib.Path(__file__).parents[1] / 'cache' / f"{pathlib.Path(a.strat).stem}_{pathlib.Path(a.tape).stem}.npz")
np.savez(out, x=np.array(X), tk=np.array(TK, dtype=float), od=np.array(OD, dtype=float), fl=np.array(FL, dtype=float),
         curve=np.array(curve))
print(out, len(X), len(TK), len(OD), len(FL))
