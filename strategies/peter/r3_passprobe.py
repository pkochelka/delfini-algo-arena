"""Passive-fill anatomy for r3_man on one tape: markout by queue position at fill, taker event size, order age, regime.

    uv run python strategies/peter/r3_passprobe.py [data/round3_public.csv] ['{"k":v}']
"""
import json, sys, bisect
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1])); sys.path.insert(0, str(HERE))


def main():
    from arena import load_agent
    from simulator.main import MAX_POSITION, MAX_POTENTIAL_POSITION, ORDER_ACCEPT_DELAY_MS, POSITION_BREACH_PENALTY_PER_SECOND
    from simulator.models import OrderStatus, OrderType
    from simulator.simulator import ExchangeSimulator, SimulationConfig
    tape = sys.argv[1] if len(sys.argv) > 1 else 'data/round3_public.csv'
    over = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    s = load_agent(HERE / 'r3_man.py', over, 'pp')
    meta = {}  # oid -> (t_sent, mx_at_send, side, price, F, turb)
    last = {'snap': None}
    mids, fills, evq = [], [], {}
    so0 = s.send_order
    def so(req):
        oid = so0(req)
        if req.order_type is OrderType.LIMIT:
            meta[oid] = (s.now_t, s.max_id, req.side.name, req.price, s.f.mean if s.f else 0, s.turb)
        return oid
    s.send_order = so
    onb0, onp0 = s.on_order_book_change, s.on_private_order_update
    def onb(snap, trades):
        if trades:
            evq[snap.timestamp_us] = sum(tr.quantity for tr in trades if tr.taker_owner == 'market')
        last['snap'] = snap
        onb0(snap, trades)
        if snap.bids and snap.asks and (not mids or snap.timestamp_us - mids[-1][0] > 20_000):
            mids.append((snap.timestamp_us, (snap.bids[0].price + snap.asks[0].price) / 2, s.f.mean if s.f else 0))
    def onp(ev):
        m = meta.get(ev.order_id)
        if m and ev.status in (OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED) and ev.filled_quantity:
            sn = last['snap']
            views = sn.bids if m[2] == 'BUY' else sn.asks
            ahead = sum(v.quantity for v in views if v.price == m[3] and v.owner == 'market' and v.order_id <= m[1])
            tot = sum(v.quantity for v in views if v.price == m[3] and v.owner == 'market')
            best = views[0].price if views else m[3]
            fills.append((s.now_t, 1 if m[2] == 'BUY' else -1, ev.filled_quantity, ev.fill_price, ahead, tot,
                          s.now_t - m[0], int(m[5]), abs(best - m[3]), s.f.mean, s.pos))
        onp0(ev)
    s.on_order_book_change, s.on_private_order_update = onb, onp
    cfg = SimulationConfig(max_position=MAX_POSITION, max_potential_position=MAX_POSITION * 10,
                           position_breach_penalty_per_second=POSITION_BREACH_PENALTY_PER_SECOND, order_accept_delay_ms=ORDER_ACCEPT_DELAY_MS)
    r = ExchangeSimulator(strategy=s, config=cfg).run(Path(tape))
    M = np.array(mids); mt, mv, mf = M[:, 0] / 1e6, M[:, 1], M[:, 2]
    F = np.array(fills)
    t, sg, q, px, ahead, tot, age, turb, dbest, Ff, pos = F.T
    at = lambda arr, tt: arr[np.clip(np.searchsorted(mt, tt, 'right') - 1, 0, len(mt) - 1)]
    print(f'pnl {r.pnl:+.0f}  passive fills {len(F)} lots {q.sum():.0f}')
    mo = {h: sg * (at(mv, t + h) - px) for h in (0.1, 1, 5, 20)}
    edge0 = sg * (Ff - px)
    def rep(name, mask):
        if mask.sum() == 0:
            return
        w = q[mask]
        print(f'  {name:26} lots {w.sum():8.0f} ({w.sum()/q.sum():.2f})  edgeF {np.average(edge0[mask], weights=w):+.2f}  ' +
              '  '.join(f'mo{h}s {np.average(mo[h][mask], weights=w):+.2f}' for h in mo) + f'  tot1s {np.sum(mo[1][mask]*w)/1e3:+.0f}k')
    rep('all', q > 0)
    for reg in (0, 1):
        rep(f'regime {"turb" if reg else "calm"}', turb == reg)
    print(' by queue ahead at fill (market lots ahead):')
    for lo, hi in [(0, 1), (1, 100), (100, 400), (400, 1000), (1000, 1e9)]:
        rep(f'  ahead [{lo},{hi})', (ahead >= lo) & (ahead < hi))
    print(' by market taker event size:')
    tq = np.array([evq.get(int(round(x * 1e6)), 0) for x in t])
    for lo, hi in [(0, 30), (30, 100), (100, 300), (300, 1000), (1000, 1e9)]:
        rep(f'  taker q [{lo},{hi})', (tq >= lo) & (tq < hi))
    print(' by order age at fill (s):')
    for lo, hi in [(0, .01), (.01, .1), (.1, .5), (.5, 2), (2, 1e9)]:
        rep(f'  age [{lo},{hi})', (age >= lo) & (age < hi))
    print(' by our price vs same-side touch at fill (ticks behind):')
    for v in (0, 1, 2):
        rep(f'  behind {v}', dbest == v)
    print(' by edge vs F at fill:')
    for lo, hi in [(-9, 0), (0, 0.3), (0.3, 0.6), (0.6, 1.0), (1.0, 9)]:
        rep(f'  edgeF [{lo},{hi})', (edge0 >= lo) & (edge0 < hi))
    print(' by |pos| before fill reduces vs adds:')
    adds = sg * (pos - sg * q) >= 0
    rep('  adds to |pos|', adds); rep('  reduces |pos|', ~adds)
    np.save(HERE.parents[1] / 'logs/r3man/passfills.npy', np.column_stack([F, tq]))


if __name__ == '__main__':
    main()
