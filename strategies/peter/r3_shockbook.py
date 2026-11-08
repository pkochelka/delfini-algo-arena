"""Record the public book around every shock jump (passive probe, no orders) -> logs/r3man/shockbook_<tape>.pkl

    uv run python strategies/peter/r3_shockbook.py data/round3_public.csv
Each shock: list of (t_us, bids[(price, qty, owner, id)], asks[...], trades[(price, qty, book_side, taker_owner)]) for
snapshots in [t_jump - 0.3 s, t_jump + 0.5 s]. Jump times from shock_events.csv (real) or .truth.json onsets + 3.0 (synth).
"""
from __future__ import annotations
import csv, json, pickle, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from simulator.strategy import BaseStrategy  # noqa: E402


def jump_times(tape: Path):
    tj = tape.with_suffix('.truth.json')
    if tj.exists():
        tr = json.loads(tj.read_text())
        return [o + 3.0 for o in sorted(tr['onsets'])], tr.get('jumps')
    rows = list(csv.DictReader(open(ROOT / 'marek/research/round3/flow/shock_events.csv')))
    return [float(r['t_jump']) for r in rows], [float(r['jump']) for r in rows]


class Probe(BaseStrategy):
    def __init__(self, tj):
        super().__init__()
        self.tj = tj; self.i = 0; self.rec = [[] for _ in tj]

    def on_order_book_change(self, snap, trades):
        t = snap.timestamp_us / 1e6
        while self.i < len(self.tj) and t > self.tj[self.i] + 0.5:
            self.i += 1
        if self.i < len(self.tj) and t >= self.tj[self.i] - 0.3:
            self.rec[self.i].append((snap.timestamp_us, [(v.price, v.quantity, v.owner, v.order_id) for v in snap.bids],
                                     [(v.price, v.quantity, v.owner, v.order_id) for v in snap.asks],
                                     [(tr.price, tr.quantity, tr.book_side, tr.taker_owner) for tr in trades]))

    def on_private_order_update(self, ev):
        pass


if __name__ == '__main__':
    from simulator.simulator import ExchangeSimulator, SimulationConfig
    tape = Path(sys.argv[1])
    tj, J = jump_times(tape)
    pr = Probe(tj)
    ExchangeSimulator(strategy=pr, config=SimulationConfig()).run(tape)
    out = ROOT / f'logs/r3man/shockbook_{tape.stem}.pkl'
    pickle.dump({'tj': tj, 'J': J, 'rec': pr.rec}, open(out, 'wb'))
    print(out, [len(r) for r in pr.rec])
