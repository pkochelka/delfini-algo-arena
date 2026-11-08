"""Record every unique stale market order beyond the take threshold, with the room we had when first seen."""
import sys, importlib.util, math
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[5]; sys.path.insert(0, str(ROOT))
from simulator.main import (MAX_POSITION, MAX_POTENTIAL_POSITION, ORDER_ACCEPT_DELAY_MS, POSITION_BREACH_PENALTY_PER_SECOND)
from simulator.simulator import ExchangeSimulator, SimulationConfig
spec = importlib.util.spec_from_file_location('m', sys.argv[1]); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
class P(m.FVAShock):
    def __init__(s):
        super().__init__(); s.seen=set(); s.rec=[]; s.path=[]; s.nxt=0.0
    def _quote(s, f, bb, ba, bids, asks):
        p = s.p; F, sd = f.mean, f.sd
        turb = s.turb
        take, tsd = (p['take_t'], p['take_sd_t']) if turb else (p['take'], p['take_sd'])
        tk = take + tsd*sd
        rest_b = sum(o[2] for o in s.orders.values() if o[0] is m.BUY); rest_a = sum(o[2] for o in s.orders.values() if o[0] is m.SELL)
        room_b = 100 - s.pos - rest_b; room_a = 100 + s.pos - rest_a
        for v in asks:
            if v.price > F - tk: break
            if v.owner == 'market' and v.order_id not in s.seen:
                s.seen.add(v.order_id); s.rec.append((s.now_t, 1, v.price, v.quantity, F, sd, room_b, int(turb), s.pos))
        for v in bids:
            if v.price < F + tk: break
            if v.owner == 'market' and v.order_id not in s.seen:
                s.seen.add(v.order_id); s.rec.append((s.now_t, -1, v.price, v.quantity, F, sd, room_a, int(turb), s.pos))
        super()._quote(f, bb, ba, bids, asks)
    def on_order_book_change(s, snap, tr):
        super().on_order_book_change(snap, tr)
        if s.f is not None and s.now_t >= s.nxt:
            s.path.append((s.now_t, s.f.mean)); s.nxt = s.now_t + 0.05
cfg = SimulationConfig(max_position=MAX_POSITION, max_potential_position=MAX_POTENTIAL_POSITION, position_breach_penalty_per_second=POSITION_BREACH_PENALTY_PER_SECOND, order_accept_delay_ms=ORDER_ACCEPT_DELAY_MS)
st = P(); r = ExchangeSimulator(strategy=st, config=cfg).run(Path(sys.argv[2]))
np.savez(sys.argv[3], rec=np.array(st.rec), path=np.array(st.path)); print(r.pnl)
