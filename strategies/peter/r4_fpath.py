"""Record the round-3 S filter's causal path (r3_dev_best's FV) on a tape, without trading.

    uv run python strategies/peter/r4_fpath.py data/round4_public.csv [--out logs/r4/fpath_round4.npz]

Writes one row per market event: t_us, F (filter mean), sd, P(volatile), transient stock Y, market touch, and the
filled taker quantity signed by aggressor (0 for non-trading events). Used by r4_dyn.py to measure how the hidden
price moves on round 4 (drift between takers, impact propagator, informed takers).
"""
from __future__ import annotations

import argparse
import sys
from array import array
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from fastsim.engine import load_strategy, run_backtest  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('tape')
    ap.add_argument('--strategy', default='strategies/marek/round3/r3_dev_best.py')
    ap.add_argument('--out')
    a = ap.parse_args()
    strat = load_strategy(ROOT / a.strategy, {'trade_start_us': 10**15})
    cols = {k: array('d') for k in ('t', 'F', 'sd', 'pv', 'Y', 'bb', 'ba', 'q')}
    base = strat.__class__.on_order_book_change

    def rec(snapshot, trades):
        base(strat, snapshot, trades)
        f = strat.f
        if f is None:
            return
        q = 0
        for tr in trades:
            if tr.taker_owner == 'market':
                q += tr.quantity if tr.book_side == 'ask' else -tr.quantity
        b, s = snapshot.bids, snapshot.asks
        cols['t'].append(snapshot.timestamp_us)
        cols['F'].append(f.mean)
        cols['sd'].append(f.sd)
        cols['pv'].append(f.pv)
        cols['Y'].append(f.Y)
        cols['bb'].append(b[0].price if b else np.nan)
        cols['ba'].append(s[0].price if s else np.nan)
        cols['q'].append(q)

    strat.on_order_book_change = rec
    run_backtest(strat, ROOT / a.tape)
    out = Path(a.out or ROOT / 'logs' / 'r4' / f'fpath_{Path(a.tape).stem}.npz')
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **{k: np.frombuffer(v, dtype=float) for k, v in cols.items()})
    print('wrote', out, len(cols['t']))


if __name__ == '__main__':
    main()
