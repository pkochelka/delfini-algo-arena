"""Record a strategy's filter state, book, position and taker flow every 100 ms over one tape (for r4_alpha.py).

    uv run python strategies/peter/r4_record.py STRAT.py data/round4_public.csv logs/r4/rec_devbest.npz ['{"k":1}']
Columns: t, F, sd, pv, exc, bb, ba, pos, turb, cum signed taker qty, cum taker count signed, jumps, cum cash,
n_shock (sg_t set), pnl (cash + pos * mid).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fastsim.engine import FastSimulator, grader_config, load_strategy  # noqa: E402


def main() -> None:
    strat_path, tape, out = sys.argv[1:4]
    over = json.loads(sys.argv[4]) if len(sys.argv) > 4 else {}
    s = load_strategy(strat_path, over)
    cls = type(s)
    rows: list = []
    st = {'next': 0.0, 'q': 0.0, 'n': 0.0, 'cash': 0.0}
    orig_book = cls.on_order_book_change
    orig_priv = cls.on_private_order_update

    def book(self, snap, trades):
        for tr in trades:
            if tr.taker_owner == 'market':
                sg = 1 if tr.book_side == 'ask' else -1
                st['q'] += sg * tr.quantity
                st['n'] += sg
        orig_book(self, snap, trades)
        t = snap.timestamp_us * 1e-6
        if t >= st['next'] and self.f is not None and snap.bids and snap.asks:
            st['next'] = (int(t * 10) + 1) / 10
            f = self.f
            bb, ba = snap.bids[0].price, snap.asks[0].price
            rows.append((t, f.mean, f.sd, f.pv, f.exc, bb, ba, self.pos, float(self.turb), st['q'], st['n'],
                         f.jumps, st['cash'], float(self.sg_t is not None), st['cash'] + self.pos * (bb + ba) / 2))

    def priv(self, ev):
        o = self.orders.get(ev.order_id)
        if o is not None and ev.filled_quantity and ev.fill_price is not None:
            st['cash'] += (-1 if o[0].name == 'BUY' else 1) * ev.filled_quantity * ev.fill_price
        orig_priv(self, ev)

    cls.on_order_book_change = book
    cls.on_private_order_update = priv
    sim = FastSimulator(s, grader_config())
    r = sim.run(ROOT / tape)
    a = np.array(rows)
    np.savez_compressed(out, a=a, cols=np.array('t F sd pv exc bb ba pos turb cq cn jumps cash shock pnl'.split()))
    print(json.dumps({'pnl': r.pnl, 'rows': len(rows), 'breach': r.breach_penalty}))


if __name__ == '__main__':
    main()
