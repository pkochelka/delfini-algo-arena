"""Trace our orders/fills and book touch in a time window. trace.py STRAT TAPE T0 T1 [k=v ...]"""
import sys, pathlib, ast
ROOT = pathlib.Path(__file__).resolve().parents[5]; sys.path.insert(0, str(ROOT))
from fastsim.engine import load_strategy, run_backtest
t0, t1 = float(sys.argv[3]), float(sys.argv[4])
over = {k: ast.literal_eval(v) for k, v in (kv.split('=', 1) for kv in sys.argv[5:])}
s = load_strategy(sys.argv[1], over)
T = [0.0]; base = [None]
ob, osend, opriv = s.on_order_book_change, s.send_order, s.on_private_order_update
def hook(snap, trades):
    if base[0] is None: base[0] = snap.timestamp_us * 1e-6
    T[0] = snap.timestamp_us * 1e-6 - base[0]
    pre = (s.f.mean if s.f else None)
    ob(snap, trades)
    if t0 <= T[0] <= t1 and snap.bids and snap.asks:
        tr = [(x.price, x.quantity, x.book_side, x.taker_owner) for x in trades]
        print(f'{T[0]:.4f} bb {snap.bids[0].price} ba {snap.asks[0].price} F {s.f.mean:.2f} sd {s.f.sd:.2f} pos {s.pos} sg {s.sg_t is not None} dir {getattr(s,"ss_dir",0)} {tr if tr else ""}')
def snd(o):
    if t0 <= T[0] <= t1: print(f'   SEND {o.side} {o.price} x{o.quantity} {o.order_type}')
    return osend(o)
def prv(e):
    if t0 <= T[0] <= t1 and e.filled_quantity: print(f'   FILL {e.status} {e.filled_quantity}')
    return opriv(e)
s.on_order_book_change = hook; s.send_order = snd; s.on_private_order_update = prv
run_backtest(s, ROOT / sys.argv[2])
