"""Run a strategy on a backtest, recording every fill with book state; then markouts at +1 s / +5 s.

uv run python strategies/peter/fill_probe.py strategy.py data.csv out.npz [json-overrides]
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, 'strategies/peter')
from arena import load_agent
from simulator.main import (MAX_POSITION, MAX_POTENTIAL_POSITION, ORDER_ACCEPT_DELAY_MS,
                            POSITION_BREACH_PENALTY_PER_SECOND)
from simulator.models import OrderStatus, Side
from simulator.simulator import ExchangeSimulator, SimulationConfig

strat_path, data, out = sys.argv[1:4]
ov = json.loads(sys.argv[4]) if len(sys.argv) > 4 else {}
s = load_agent(Path(strat_path), ov, 'probe')
mids, fills = [], []
qinfo = {}; born = {}
state = {'qb': 0, 'qa': 0, 'bb': 0, 'ba': 0, 't': 0}
orig_book, orig_priv = s.on_order_book_change, s.on_private_order_update

def mkt_touch(orders):
    p, q = None, 0
    for o in orders:
        if o.owner != 'market': continue
        if p is None: p = o.price
        elif o.price != p: break
        q += o.quantity
    return p, q

def book(snap, trades):
    (bb, qb), (ba, qa) = mkt_touch(snap.bids), mkt_touch(snap.asks)
    if bb is not None and ba is not None:
        state.update(qb=qb, qa=qa, bb=bb, ba=ba, t=snap.timestamp_us)
        mids.append((snap.timestamp_us, (bb + ba) / 2))
    for lst in (snap.bids, snap.asks):
        ahead = 0; lvl = None; tot = {}
        for o in lst:
            tot[o.price] = tot.get(o.price, 0) + o.quantity
        for o in lst:
            if o.price != lvl: lvl = o.price; ahead = 0
            if o.owner == 'user': qinfo[o.order_id] = (ahead, tot[o.price])
            ahead += o.quantity
    orig_book(snap, trades)

def priv(ev):
    if ev.status in (OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED) and ev.filled_quantity:
        o = s.orders.get(ev.order_id) if hasattr(s, 'orders') else None
        side = o[0] if o else None
        sg = 1 if side == Side.BUY else -1
        ah, tot = qinfo.get(ev.order_id, (-1, -1))
        age = state['t'] - born.get(ev.order_id, state['t'])
        fills.append((state['t'], sg, ev.filled_quantity, ev.fill_price, state['qb'], state['qa'], state['bb'], state['ba'], getattr(s, 'pos', 0), ah, tot, age))
    if ev.status == OrderStatus.ACCEPTED: born[ev.order_id] = state['t']
    orig_priv(ev)

s.on_order_book_change, s.on_private_order_update = book, priv
cfg = SimulationConfig(max_position=MAX_POSITION, max_potential_position=MAX_POTENTIAL_POSITION,
                       position_breach_penalty_per_second=POSITION_BREACH_PENALTY_PER_SECOND,
                       order_accept_delay_ms=ORDER_ACCEPT_DELAY_MS)
r = ExchangeSimulator(strategy=s, config=cfg).run(Path(data))
m = np.array(mids); f = np.array(fills, dtype=float)
np.savez(out, mids=m, fills=f)
print(f'pnl={r.pnl:.0f} fills={r.fills} vol={r.traded_volume} q50={r.abs_inventory_q50} q95={r.abs_inventory_q95}')
