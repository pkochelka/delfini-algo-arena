import sys, importlib.util
from pathlib import Path
ROOT = Path(__file__).resolve().parents[5]; sys.path.insert(0, str(ROOT))
from simulator.main import (MAX_POSITION, MAX_POTENTIAL_POSITION, ORDER_ACCEPT_DELAY_MS, POSITION_BREACH_PENALTY_PER_SECOND)
from simulator.simulator import ExchangeSimulator, SimulationConfig
spec = importlib.util.spec_from_file_location('m', sys.argv[1]); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
Js = [float(x) for x in sys.argv[3].split(',')]
class P(m.FVAShock):
    def on_order_book_change(s, snap, tr):
        pmax = s.max_id; pbb, pba = s.mbb, s.mba
        super().on_order_book_change(snap, tr)
        t = s.now_t
        for J in Js:
            if J - 0.15 <= t <= J + 0.12:
                new = ''
                if s.max_id != pmax:
                    for v in snap.bids:
                        if v.order_id == s.max_id: new = f'NEW bid {v.price} q{v.quantity}'
                    for v in snap.asks:
                        if v.order_id == s.max_id: new = f'NEW ask {v.price} q{v.quantity}'
                trs = [(x.price, x.quantity, x.book_side, x.taker_owner) for x in tr]
                if new or trs or s.ss_dir:
                    print(f'{J:.2f} {t-J:+.4f} prev {pbb}/{pba} now {s.mbb}/{s.mba} F{s.f.mean:.1f} pos{s.pos} ssdir{s.ss_dir} onset{(s.ss_onset or 0)-J+3:+.3f} {new} {trs}')
cfg = SimulationConfig(max_position=MAX_POSITION, max_potential_position=MAX_POTENTIAL_POSITION, position_breach_penalty_per_second=POSITION_BREACH_PENALTY_PER_SECOND, order_accept_delay_ms=ORDER_ACCEPT_DELAY_MS)
ExchangeSimulator(strategy=P(), config=cfg).run(Path(sys.argv[2]))
