"""Probe fvs_mm-style strategy: record fills with context; save npz."""
import sys, json, importlib.util, argparse
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from simulator.main import (MAX_POSITION, MAX_POTENTIAL_POSITION, ORDER_ACCEPT_DELAY_MS, POSITION_BREACH_PENALTY_PER_SECOND)
from simulator.models import OrderStatus, OrderType, Side
from simulator.simulator import ExchangeSimulator, SimulationConfig

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('strategy'); ap.add_argument('tape'); ap.add_argument('out')
    ap.add_argument('--set', nargs='*', default=[])
    a = ap.parse_args()
    spec = importlib.util.spec_from_file_location('stratmod', a.strategy); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    import ast
    for kv in a.set:
        k, v = kv.split('=', 1); m.PARAMS[k] = ast.literal_eval(v)
    class Probe(m.FVAShock):
        def __init__(s):
            super().__init__(); s.kind = {}; s.age = {}; s.fills = []; s.path = []; s.nxt = 0.0
            s.last_tk_t = -1.0; s.last_tk_q = 0; s.cash = 0.0; s.ahead = {}
        def _send(s, side, price, qty, kind):
            super()._send(side, price, qty, kind)
            oid = next(reversed(s.orders))
            s.kind[oid] = (1 if kind is OrderType.IOC else 0) + (2 if s.ss_dir else 0)
            s.age[oid] = s.now_t
        def on_private_order_update(s, e):
            o = s.orders.get(e.order_id)
            if o is not None and e.status in (OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED):
                f = s.f; sg = 1 if o[0] is Side.BUY else -1
                s.cash -= sg * e.filled_quantity * e.fill_price
                s.fills.append((e.timestamp_us / 1e6, sg, e.filled_quantity, e.fill_price, s.kind.get(e.order_id, 0),
                                int(s.turb), f.mean, f.sd, s.now_t - s.last_tk_t, s.last_tk_q, e.timestamp_us/1e6 - s.age.get(e.order_id, 0), s.pos, o[1], s.ahead.get(e.order_id, -1)))
            super().on_private_order_update(e)
        def on_order_book_change(s, snap, tr):
            q = sum(t.quantity for t in tr if t.taker_owner == 'market')
            super().on_order_book_change(snap, tr)
            if s.orders:
                for views in (snap.bids, snap.asks):
                    acc = 0
                    for v in views:
                        if v.owner == 'market': acc += v.quantity
                        elif v.order_id in s.orders: s.ahead[v.order_id] = acc
            if q:
                s.last_tk_t = s.now_t; s.last_tk_q = q
            if s.f is not None and s.now_t >= s.nxt:
                s.path.append((s.now_t, s.f.mean, int(s.turb), s.pos, s.cash + s.pos * s.f.mean, s.f.pv, (s.sg_t or 0), (s.ss_onset or 0), s.n_snipes, s.n_shocks))
                s.nxt = s.now_t + 0.05
    st = Probe()
    cfg = SimulationConfig(max_position=MAX_POSITION, max_potential_position=MAX_POTENTIAL_POSITION,
                           position_breach_penalty_per_second=POSITION_BREACH_PENALTY_PER_SECOND, order_accept_delay_ms=ORDER_ACCEPT_DELAY_MS)
    r = ExchangeSimulator(strategy=st, config=cfg).run(Path(a.tape))
    np.savez(a.out, fills=np.array(st.fills), path=np.array(st.path))
    print(a.tape, r.pnl, r.breach_penalty)
main()
