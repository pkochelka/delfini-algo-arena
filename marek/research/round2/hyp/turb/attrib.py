"""Attribution of turb.py fills by regime kind tk (0 calm,1 burst,2 vol,3 other,4 shock) and passive/take."""
import argparse, json, sys, ast
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
ROOT = Path('/Users/marek/Documents/delfini-algo-arena')
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'strategies/marek/round2/hyp'))
TAPES = {'r2': 'data/round2.csv', **{f's{i}': f'data/synth/r2_v2_seed{i}.csv' for i in range(1, 8)}}

def run(tape, over):
    import turb as M
    from simulator.main import (MAX_POSITION, MAX_POTENTIAL_POSITION, ORDER_ACCEPT_DELAY_MS, POSITION_BREACH_PENALTY_PER_SECOND)
    from simulator.models import OrderStatus, OrderType
    from simulator.simulator import ExchangeSimulator, SimulationConfig
    M.PARAMS.update(over)
    class Probe(M.FVAShock):
        def __init__(s):
            super().__init__(); s.kind = {}; s.fills = []; s.path = []; s.nxt = 0.0; s.tk = 0
        def _send(s, side, price, qty, kind):
            super()._send(side, price, qty, kind)
            s.kind[next(reversed(s.orders))] = (1 if kind is OrderType.IOC else 0, s.tk if s.turb else 0)
        def on_private_order_update(s, e):
            o = s.orders.get(e.order_id)
            if o is not None and e.status in (OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED):
                k, tk = s.kind.get(e.order_id, (1, 5))
                s.fills.append((e.timestamp_us / 1e6, 1 if o[0] is M.BUY else -1, e.filled_quantity, e.fill_price, k, tk))
            super().on_private_order_update(e)
        def on_order_book_change(s, snap, tr):
            super().on_order_book_change(snap, tr)
            if s.f is not None and s.now_t >= s.nxt:
                s.path.append((s.now_t, s.f.mean, s.tk if s.turb else 0, abs(s.pos))); s.nxt = s.now_t + 0.1
    st = Probe()
    cfg = SimulationConfig(max_position=MAX_POSITION, max_potential_position=MAX_POTENTIAL_POSITION,
                           position_breach_penalty_per_second=POSITION_BREACH_PENALTY_PER_SECOND, order_accept_delay_ms=ORDER_ACCEPT_DELAY_MS)
    r = ExchangeSimulator(strategy=st, config=cfg).run(ROOT / TAPES[tape])
    fl = np.array(st.fills); pt = np.array(st.path)
    out = {'tape': tape, 'pnl': r.pnl, 'share': {int(k): round(float((pt[:, 2] == k).mean()), 3) for k in range(5)}}
    out['pos90'] = {int(k): round(float((pt[pt[:, 2] == k, 3] >= 90).mean()), 3) for k in range(5) if (pt[:, 2] == k).any()}
    out['posmean'] = {int(k): round(float(pt[pt[:, 2] == k, 3].mean()), 1) for k in range(5) if (pt[:, 2] == k).any()}
    for H in (1.0, 10.0):
        j = np.minimum(np.searchsorted(pt[:, 0], fl[:, 0] + H), len(pt) - 1)
        m_ = fl[:, 1] * (pt[j, 1] - fl[:, 3]) * fl[:, 2]
        for kind in (0, 1):
            for tk in range(6):
                m = (fl[:, 4] == kind) & (fl[:, 5] == tk)
                if not m.any(): continue
                key = f"{'take' if kind else 'pass'}_{tk}"
                out.setdefault(key, {})['lots'] = int(fl[m, 2].sum())
                out[key][f'tot{H:g}'] = round(float(m_[m].sum()))
    return out

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--tapes', default='r2,s1'); ap.add_argument('--set', nargs='*', default=[])
    a = ap.parse_args()
    over = {k: ast.literal_eval(v) for k, v in (x.split('=', 1) for x in a.set)}
    tp = a.tapes.split(',')
    with ProcessPoolExecutor(len(tp)) as ex:
        for r in ex.map(run, tp, [over] * len(tp)):
            print(r['tape'], 'pnl', r['pnl'], 'share', r['share'])
            for k, v in r.items():
                if isinstance(v, dict) and k not in ('share',): print('   ', k, v)
