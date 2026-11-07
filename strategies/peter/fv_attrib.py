"""PnL attribution for fv_mm: every fill split by kind (passive / take) and regime (calm / turbulent),
with markouts against the filter's S at +1 s and +10 s (positive = good for us).

    uv run python strategies/peter/fv_attrib.py data/splits_r2/wf/wf06.csv data/splits_r2/wf/wf03.csv [--set k=v ...]
"""
from __future__ import annotations

import argparse, json, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))


def run(path: str, over: dict, trade_start_us: int) -> dict:
    import fv_mm
    from simulator.main import (MAX_POSITION, MAX_POTENTIAL_POSITION, ORDER_ACCEPT_DELAY_MS,
                                POSITION_BREACH_PENALTY_PER_SECOND)
    from simulator.models import OrderStatus, OrderType
    from simulator.simulator import ExchangeSimulator, SimulationConfig

    fv_mm.PARAMS = fv_mm.PARAMS | over | {'trade_start_us': trade_start_us}

    class Probe(fv_mm.FVMM):
        def __init__(self):
            super().__init__()
            self.kind = {}
            self.fills = []  # t, sign, qty, price, kind(0 passive, 1 take), turb, F
            self.path = []  # t, F (every 100 ms)
            self.nxt = 0.0

        def _send(self, side, price, qty, kind):
            super()._send(side, price, qty, kind)
            self.kind[next(reversed(self.orders))] = 1 if kind is OrderType.IOC else 0

        def on_private_order_update(self, e):
            o = self.orders.get(e.order_id)
            if o is not None and e.status in (OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED):
                f = self.f
                self.fills.append((e.timestamp_us / 1e6, 1 if o[0] is fv_mm.BUY else -1, e.filled_quantity,
                                   e.fill_price, self.kind.get(e.order_id, 0), int(self.turb), f.mean))
            super().on_private_order_update(e)

        def on_order_book_change(self, s, tr):
            super().on_order_book_change(s, tr)
            if self.f is not None and self.now_t >= self.nxt:
                self.path.append((self.now_t, self.f.mean, int(self.turb)))
                self.nxt = self.now_t + 0.1

    st = Probe()
    cfg = SimulationConfig(max_position=MAX_POSITION, max_potential_position=MAX_POTENTIAL_POSITION,
                           position_breach_penalty_per_second=POSITION_BREACH_PENALTY_PER_SECOND,
                           order_accept_delay_ms=ORDER_ACCEPT_DELAY_MS)
    r = ExchangeSimulator(strategy=st, config=cfg).run(Path(path))
    fl = np.array(st.fills); pt = np.array(st.path)
    out = {'fold': Path(path).stem, 'pnl': r.pnl, 'turb_share': float(pt[pt[:, 0] >= trade_start_us / 1e6, 2].mean())}
    for H in (1.0, 10.0):
        j = np.minimum(np.searchsorted(pt[:, 0], fl[:, 0] + H), len(pt) - 1)
        fl_m = fl[:, 1] * (pt[j, 1] - fl[:, 3]) * fl[:, 2]  # signed edge vs future S, per fill
        for kind in (0, 1):
            for tb in (0, 1):
                m = (fl[:, 4] == kind) & (fl[:, 5] == tb)
                key = f"{'take' if kind else 'pass'}_{'turb' if tb else 'calm'}"
                out.setdefault(key, {})['lots'] = int(fl[m, 2].sum())
                out[key][f'mk{H:g}'] = round(float(fl_m[m].sum() / max(1, fl[m, 2].sum())), 3)
                out[key][f'tot{H:g}'] = round(float(fl_m[m].sum()))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('files', nargs='+'); ap.add_argument('--set', nargs='*', default=[])
    a = ap.parse_args()
    over = {k: json.loads(v) for k, v in (x.split('=') for x in a.set)}
    folds = json.loads((Path(a.files[0]).parent / 'folds.json').read_text())['folds']
    ts = {f['name']: f['trade_start_us'] for f in folds}
    with ProcessPoolExecutor(len(a.files)) as ex:
        for r in ex.map(run, a.files, [over] * len(a.files), [ts.get(Path(f).stem, 0) for f in a.files]):
            print(r['fold'], f"pnl {r['pnl']:+.0f} turb_share {r['turb_share']:.2f}")
            for k, v in r.items():
                if isinstance(v, dict):
                    print(f'   {k:10} {v}')


if __name__ == '__main__':
    main()
