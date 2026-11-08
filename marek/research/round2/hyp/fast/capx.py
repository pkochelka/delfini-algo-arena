import sys, importlib.util
from pathlib import Path
ROOT = Path(__file__).resolve().parents[5]; sys.path.insert(0, str(ROOT))
from simulator.simulator import ExchangeSimulator, SimulationConfig
spec = importlib.util.spec_from_file_location('m', sys.argv[1]); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
cap = int(sys.argv[3]); m.PARAMS.update(cap=cap, cap_t=cap)
cfg = SimulationConfig(max_position=100000, max_potential_position=100000, position_breach_penalty_per_second=0.0, order_accept_delay_ms=5)
r = ExchangeSimulator(strategy=m.FVAShock(), config=cfg).run(Path(sys.argv[2]))
print('cap', cap, r.pnl, 'absq95', r.abs_inventory_q95)
