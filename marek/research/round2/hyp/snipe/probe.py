"""Run a strategy file in-process on a full tape; dump fills (tagged), equity trace, snipe triggers, breaches.
uv run python marek/research/round2/hyp/snipe/probe.py STRAT.py TAPE OUT.pkl key=val ..."""
import sys, ast, pathlib, importlib.util, pickle
ROOT = pathlib.Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
strat, tape, out = sys.argv[1], sys.argv[2], sys.argv[3]
over = {k: ast.literal_eval(v) for k, v in (a.split('=', 1) for a in sys.argv[4:])}
spec = importlib.util.spec_from_file_location('m', ROOT / strat)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
m.PARAMS.update(over)
from simulator.simulator import ExchangeSimulator, SimulationConfig
from simulator.main import MAX_POSITION, MAX_POTENTIAL_POSITION, POSITION_BREACH_PENALTY_PER_SECOND, ORDER_ACCEPT_DELAY_MS
from simulator.models import OrderType
TAPES = {'r2': 'data/round2.csv', **{f's{i}': f'data/synth/r2_v2_seed{i}.csv' for i in range(1, 8)}}
fills, eq, trig, breaches = [], [], [], []
tags = {}
class P(m.FVAShock):
    def __init__(self):
        super().__init__()
        self.cash = 0.0
        self._next = 0.0
        self._bt = None
        self._nsn = 0
    def send_order(self, req):
        oid = super().send_order(req)
        if req.order_type is OrderType.IOC and self.ss_dir and self.ss_left > 0 and not getattr(self, '_in_xu', False):
            tags[oid] = 'ss'
        elif getattr(self, '_in_xu', False):
            tags[oid] = 'xu'
        return oid
    def _xunwind(self, snapshot):
        self._in_xu = True
        try:
            super()._xunwind(snapshot)
        finally:
            self._in_xu = False
    def on_private_order_update(self, e):
        o = self.orders.get(e.order_id)
        b = abs(self.pos) > 100
        if o is not None and e.filled_quantity and e.fill_price is not None:
            s = 1 if o[0] is m.BUY else -1
            self.cash -= s * e.filled_quantity * e.fill_price
            fills.append((e.timestamp_us * 1e-6, s * e.filled_quantity, e.fill_price, tags.get(e.order_id, 'mm'),
                          o[1]))
        super().on_private_order_update(e)
        if not b and abs(self.pos) > 100:
            self._bt = e.timestamp_us
        if b and abs(self.pos) <= 100 and self._bt is not None:
            breaches.append((self._bt * 1e-6, (e.timestamp_us - self._bt) / 1e3))
    def on_order_book_change(self, snapshot, trades):
        super().on_order_book_change(snapshot, trades)
        if self.n_snipes != self._nsn:
            self._nsn = self.n_snipes
            trig.append((self.now_t, self.ss_dir, self.ss_ref, self.sg_t, self.f.mean if self.f else None))
        t = self.now_t
        if t >= self._next and snapshot.bids and snapshot.asks:
            self._next = t + 0.005
            eq.append((t, self.cash, self.pos, (snapshot.bids[0].price + snapshot.asks[0].price) / 2,
                       self.f.mean if self.f else 0.0))
s = P()
cfg = SimulationConfig(max_position=MAX_POSITION, max_potential_position=MAX_POTENTIAL_POSITION,
                       position_breach_penalty_per_second=POSITION_BREACH_PENALTY_PER_SECOND,
                       order_accept_delay_ms=ORDER_ACCEPT_DELAY_MS)
r = ExchangeSimulator(s, cfg).run(ROOT / TAPES[tape])
res = r.model_dump()
pickle.dump({'res': res, 'fills': fills, 'eq': eq, 'trig': trig, 'breaches': breaches}, open(out, 'wb'))
print(tape, {k: res[k] for k in ('pnl', 'breach_penalty') if k in res})
