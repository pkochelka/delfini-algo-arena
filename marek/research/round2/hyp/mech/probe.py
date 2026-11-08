"""Run mech.py in-process on a head of a tape and print breach diagnostics.
uv run python marek/research/round2/hyp/mech/probe.py TAPE NROWS key=val ..."""
import sys, ast, pathlib, itertools, importlib.util
ROOT = pathlib.Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
tape, n = sys.argv[1], int(sys.argv[2])
over = {k: ast.literal_eval(v) for k, v in (a.split('=', 1) for a in sys.argv[3:])}
spec = importlib.util.spec_from_file_location('mech', ROOT / 'strategies/marek/round2/hyp/mech.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
m.PARAMS.update(over)
from simulator.simulator import ExchangeSimulator, SimulationConfig
from simulator.main import MAX_POSITION, MAX_POTENTIAL_POSITION, POSITION_BREACH_PENALTY_PER_SECOND, ORDER_ACCEPT_DELAY_MS
src = ROOT / tape
dst = pathlib.Path('/private/tmp/claude-501/-Users-marek-Documents-delfini-algo-arena/33f1afa5-dc87-4972-b7f1-01df994e95c5/scratchpad') / f'head_{pathlib.Path(tape).stem}_{n}.csv'
if not dst.exists():
    with open(src) as f, open(dst, 'w') as g:
        g.writelines(itertools.islice(f, n + 1))
breaches = []
class P(m.FVAShock):
    def on_private_order_update(self, e):
        b = abs(self.pos) > 100
        super().on_private_order_update(e)
        if not b and abs(self.pos) > 100:
            self._bt = self.now_us()
        if b and abs(self.pos) <= 100:
            breaches.append((self._bt * 1e-6, (self.now_us() - self._bt) / 1e3))
s = P()
cfg = SimulationConfig(max_position=MAX_POSITION, max_potential_position=MAX_POTENTIAL_POSITION,
                       position_breach_penalty_per_second=POSITION_BREACH_PENALTY_PER_SECOND,
                       order_accept_delay_ms=ORDER_ACCEPT_DELAY_MS)
r = ExchangeSimulator(s, cfg).run(dst)
print(r.model_dump())
import statistics
d = [x[1] for x in breaches]
print('n breaches', len(d), 'total ms', sum(d), 'median ms', statistics.median(d) if d else None)
print('longest', sorted(breaches, key=lambda x: -x[1])[:10])
