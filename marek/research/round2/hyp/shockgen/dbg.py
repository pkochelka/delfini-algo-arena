import sys, ast, pathlib, importlib.util
ROOT = pathlib.Path(__file__).resolve().parents[5]; sys.path.insert(0, str(ROOT))
strat, tape = sys.argv[1], sys.argv[2]
over = {k: ast.literal_eval(v) for k, v in (a.split('=', 1) for a in sys.argv[3:])}
spec = importlib.util.spec_from_file_location('m', ROOT / strat); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
m.PARAMS.update(over)
from simulator.simulator import ExchangeSimulator, SimulationConfig
from simulator.main import MAX_POSITION, MAX_POTENTIAL_POSITION, POSITION_BREACH_PENALTY_PER_SECOND, ORDER_ACCEPT_DELAY_MS
TAPES = {'r2': 'data/round2.csv', **{f's{i}': f'data/synth/r2_v2_seed{i}.csv' for i in range(1, 8)}}
class P(m.FVAShock):
    def _send(self, side, price, qty, kind):
        if self.ss_dir and self.ss_left > 0:
            print('SEND %.4f %s px %d q %d ref %d F %.1f pos %d edge %.1f stk %s inw %s' % (self.now_t, side, price, qty, self.ss_ref, self.f.mean, self.pos, self.ss_edge, self.ss_stk, self.ss_inw))
        super()._send(side, price, qty, kind)
    def on_order_book_change(self, snapshot, trades):
        n = self.n_snipes; left = self.ss_left
        super().on_order_book_change(snapshot, trades)
        if self.n_snipes != n:
            rest = sum(o[2] for o in self.orders.values() if o[0] is (m.BUY if self.ss_dir > 0 else m.SELL))
            print('TRIG %.4f dir %d ref %d F %.1f pos %d rest %d edge %.1f stk %s inw %s on %s' % (self.now_t, self.ss_dir, self.ss_ref, self.f.mean, self.pos, rest, self.ss_edge, self.ss_stk, self.ss_inw, self.ss_onset))
        elif self.ss_dir and self.ss_left < left:
            print('SHOT %.4f left %d F %.1f pos %d edge %.1f' % (self.now_t, self.ss_left, self.f.mean, self.pos, self.ss_edge))
cfg = SimulationConfig(max_position=MAX_POSITION, max_potential_position=MAX_POTENTIAL_POSITION,
                       position_breach_penalty_per_second=POSITION_BREACH_PENALTY_PER_SECOND, order_accept_delay_ms=ORDER_ACCEPT_DELAY_MS)
r = ExchangeSimulator(P(), cfg).run(ROOT / TAPES[tape]); print('PNL', r.pnl)
