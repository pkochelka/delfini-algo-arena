"""PnL attribution for r3_mm / r3_man: fills by kind (passive / take / snipe / xu = excess unwind) and regime, markouts vs mid.

    uv run python strategies/peter/r3_attrib.py [--strategy strategies/peter/r3_man.py] [--dir data/splits_r3/wf | --tapes a.csv,b.csv] [--set '{"k":v}']
Sum over kinds of the 'end' column = fold PnL (marked to the final mid).
"""
from __future__ import annotations
import argparse, json, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1])); sys.path.insert(0, str(HERE))
H = (0.1, 1, 5, 20, 60)


def run(over: dict, fold: dict, strategy: str = str(HERE / 'r3_mm.py')):
    from arena import load_agent
    from simulator.main import MAX_POSITION, MAX_POTENTIAL_POSITION, ORDER_ACCEPT_DELAY_MS, POSITION_BREACH_PENALTY_PER_SECOND
    from simulator.models import OrderStatus, OrderType
    from simulator.simulator import ExchangeSimulator, SimulationConfig
    s = load_agent(Path(strategy), over | {'trade_start_us': fold['trade_start_us']}, 'attr')
    kind, mids, fills = {}, [], []
    send0, onb0, onp0 = s._send, s.on_order_book_change, s.on_private_order_update

    def send(side, price, qty, k):
        n = len(s.orders); send0(side, price, qty, k)
        oid = next(reversed(s.orders))
        tag = 'passive' if k is OrderType.LIMIT else ('snipe' if s.ss_dir and s.ss_left > 0 else 'take')
        if getattr_safe(s, 'xu_ids') is not None and oid in s.xu_ids:
            tag = 'xu'
        kind[oid] = (tag, 'turb' if s.turb else 'calm')
    def onb(snap, trades):
        onb0(snap, trades)
        if snap.bids and snap.asks and (not mids or snap.timestamp_us - mids[-1][0] > 50_000):
            mids.append((snap.timestamp_us, (snap.bids[0].price + snap.asks[0].price) / 2))
    def onp(ev):
        if ev.status in (OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED) and ev.order_id in kind:
            o = s.orders.get(ev.order_id)
            if o is not None:
                fills.append((s.now_t * 1e6, 1 if o[0].name == 'BUY' else -1, ev.filled_quantity, ev.fill_price, *kind[ev.order_id]))
        onp0(ev)
    s._send, s.on_order_book_change, s.on_private_order_update = send, onb, onp
    if hasattr(s, 'xu_ids'):  # r3_man sends the excess-unwind IOCs directly; tag them on the way out
        so0 = s.send_order
        def so(req):
            oid = so0(req)
            kind[oid] = ('xu' if s.p.get('xu_on') and abs(s.pos) > s.p['xu_tgt'] else ('snipe' if s.ss_dir else 'take'), 'turb' if s.turb else 'calm')
            return oid
        s.send_order = so
    cfg = SimulationConfig(max_position=MAX_POSITION, max_potential_position=MAX_POTENTIAL_POSITION,
                           position_breach_penalty_per_second=POSITION_BREACH_PENALTY_PER_SECOND, order_accept_delay_ms=ORDER_ACCEPT_DELAY_MS)
    r = ExchangeSimulator(strategy=s, config=cfg).run(Path(fold['path']))
    return r.pnl, mids, fills


def getattr_safe(o, name):
    return o.__dict__.get(name) if hasattr(o, '__dict__') else None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--dir', default='data/splits_r3/wf'); ap.add_argument('--set', default='{}')
    ap.add_argument('--strategy', default=str(HERE / 'r3_mm.py')); ap.add_argument('--tapes', default='')
    a = ap.parse_args()
    from tune import load_folds, parse_range
    if a.tapes:
        folds = [{'name': Path(t).stem, 'path': t, 'trade_start_us': 0} for t in a.tapes.split(',')]
    else:
        folds = load_folds(Path(a.dir), parse_range('1-12'))
    with ProcessPoolExecutor(12) as ex:
        res = list(ex.map(run, [json.loads(a.set)] * len(folds), folds, [a.strategy] * len(folds)))
    agg = {}
    for pnl, mids, fills in res:
        m = np.array(mids); mt, mv = m[:, 0], m[:, 1]
        at = lambda t: mv[np.clip(np.searchsorted(mt, t, 'right') - 1, 0, len(mv) - 1)]
        for t, sg, q, px, k, reg in fills:
            row = agg.setdefault((k, reg), np.zeros(len(H) + 3))
            row[0] += q
            for i, h in enumerate(H):
                row[1 + i] += sg * q * (at(t + h * 1e6) - px)
            row[-2] += sg * q * (mv[-1] - px)
            row[-1] += 1
    print(f"total pnl {sum(r[0] for r in res):+.0f} over {len(res)} folds")
    print(f"{'kind':8} {'reg':5} {'fills':>6} {'lots':>7} " + ' '.join(f'{f"+{h}s":>9}' for h in H) + f" {'end':>9}  (per-lot +1s, end)")
    for (k, reg), r in sorted(agg.items()):
        print(f"{k:8} {reg:5} {r[-1]:6.0f} {r[0]:7.0f} " + ' '.join(f'{v:+9.0f}' for v in r[1:-1]) + f"  {r[2]/r[0]:+.2f} {r[-2]/r[0]:+.2f}")


if __name__ == '__main__':
    main()
