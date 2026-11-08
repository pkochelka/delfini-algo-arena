"""probe inv.py: fills by kind with markouts, pos path, take log. usage: probe.py tape.csv out.npz [k=v ...]"""
import sys, ast, importlib.util
from pathlib import Path
import numpy as np
ROOT = Path('/Users/marek/Documents/delfini-algo-arena'); sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('inv', ROOT / 'strategies/marek/round2/hyp/inv.py')
inv = importlib.util.module_from_spec(spec); spec.loader.exec_module(inv)
from simulator.main import (MAX_POSITION, MAX_POTENTIAL_POSITION, ORDER_ACCEPT_DELAY_MS, POSITION_BREACH_PENALTY_PER_SECOND)
from simulator.models import OrderStatus, OrderType
from simulator.simulator import ExchangeSimulator, SimulationConfig
over = {k: ast.literal_eval(v) for k, v in (a.split('=', 1) for a in sys.argv[3:])}
inv.PARAMS.update(over)

class Probe(inv.FVAShock):
    def __init__(self):
        super().__init__(); self.kind = {}; self.fills = []; self.path = []; self.nxt = 0.0; self.tlog = []
    def _send(self, side, price, qty, kind):
        super()._send(side, price, qty, kind)
        k = 0 if kind is not OrderType.IOC else (2 if self.ss_dir else 1)
        self.kind[next(reversed(self.orders))] = k
    def on_private_order_update(self, e):
        o = self.orders.get(e.order_id)
        if o is not None and e.status in (OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED) and e.filled_quantity:
            self.fills.append((e.timestamp_us / 1e6, 1 if o[0] is inv.BUY else -1, e.filled_quantity, e.fill_price,
                               self.kind.get(e.order_id, 0), int(self.turb), self.f.mean, self.pos))
        super().on_private_order_update(e)
    def on_order_book_change(self, s, tr):
        super().on_order_book_change(s, tr)
        if self.f is not None and self.now_t >= self.nxt and s.bids and s.asks:
            self.path.append((self.now_t, self.f.mean, int(self.turb), self.pos, (s.bids[0].price + s.asks[0].price) / 2))
            self.nxt = self.now_t + 0.1

st = Probe()
cfg = SimulationConfig(max_position=MAX_POSITION, max_potential_position=MAX_POTENTIAL_POSITION,
                       position_breach_penalty_per_second=POSITION_BREACH_PENALTY_PER_SECOND, order_accept_delay_ms=ORDER_ACCEPT_DELAY_MS)
r = ExchangeSimulator(strategy=st, config=cfg).run(Path(sys.argv[1]))
print('pnl', r.pnl)
np.savez(sys.argv[2], fills=np.array(st.fills), path=np.array(st.path), tlog=np.array(st.tlog), pnl=r.pnl)
