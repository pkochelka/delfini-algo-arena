"""Per passive-fill event in calm: edge gained (lots x (F - price)) vs the carry of that same event (position after the
fill x the change of F the filling taker causes), by position before / after and fill size.

    uv run python strategies/peter/r3_fillcarry.py ['{"k":v}']
"""
import json, sys
from pathlib import Path
from collections import defaultdict
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1])); sys.path.insert(0, str(HERE))


def main():
    from arena import load_agent
    from simulator.main import MAX_POSITION, MAX_POTENTIAL_POSITION, ORDER_ACCEPT_DELAY_MS, POSITION_BREACH_PENALTY_PER_SECOND
    from simulator.models import OrderStatus, OrderType
    from simulator.simulator import ExchangeSimulator, SimulationConfig
    over = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    s = load_agent(HERE / 'r3_man.py', over, 'fc')
    pas = set()
    so0 = s.send_order
    def so(req):
        oid = so0(req)
        if req.order_type is OrderType.LIMIT:
            pas.add(oid)
        return oid
    s.send_order = so
    cur = {'f': 0, 'e': 0.0, 'p0': None, 'sg': 0}
    rows = []
    onb0, onp0 = s.on_order_book_change, s.on_private_order_update
    def onp(ev):
        if ev.order_id in pas and ev.status in (OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED) and ev.filled_quantity \
                and s.f is not None and not s.turb:
            o = s.orders.get(ev.order_id)
            if o is not None:
                sg = 1 if o[0].name == 'BUY' else -1
                if cur['p0'] is None:
                    cur['p0'] = s.pos; cur['F0'] = s.f.mean
                cur['f'] += ev.filled_quantity; cur['sg'] = sg
                cur['e'] += sg * ev.filled_quantity * (s.f.mean - ev.fill_price)
        onp0(ev)
    def onb(snap, trades):
        onb0(snap, trades)
        if cur['p0'] is not None:
            V = sum(tr.quantity for tr in trades if tr.taker_owner == 'market')
            rows.append((cur['p0'], s.pos, cur['f'], cur['e'], s.pos * (s.f.mean - cur['F0']), V))
            cur.update(f=0, e=0.0, p0=None)
    s.on_order_book_change, s.on_private_order_update = onb, onp
    cfg = SimulationConfig(max_position=MAX_POSITION, max_potential_position=MAX_POTENTIAL_POSITION,
                           position_breach_penalty_per_second=POSITION_BREACH_PENALTY_PER_SECOND, order_accept_delay_ms=ORDER_ACCEPT_DELAY_MS)
    r = ExchangeSimulator(strategy=s, config=cfg).run(Path('data/round3_public.csv'))
    print(f'pnl {r.pnl/1e3:+.1f}k  calm passive fill events {len(rows)}  edge {sum(x[3] for x in rows)/1e3:+.1f}k  event carry {sum(x[4] for x in rows)/1e3:+.1f}k')
    def grp(name, key):
        g = defaultdict(lambda: [0, 0, 0.0, 0.0])
        for p0, p1, f, e, c, V in rows:
            k = key(p0, p1, f, V)
            g[k][0] += 1; g[k][1] += f; g[k][2] += e; g[k][3] += c
        print(name)
        for k in sorted(g):
            n, f, e, c = g[k]
            print(f'   {str(k):22} events {n:6d} lots {f:7d} edge {e/1e3:+7.1f}k carry {c/1e3:+7.1f}k net {(e+c)/1e3:+7.1f}k ({(e+c)/max(f,1):+.2f}/lot)')
    grp(' by kind:', lambda p0, p1, f, V: 'adds |pos|' if abs(p1) > abs(p0) and p0 * p1 >= 0 else ('flips' if p0 * p1 < 0 else 'reduces'))
    grp(' by |pos| after:', lambda p0, p1, f, V: min(abs(p1) // 25 * 25, 100))
    grp(' by taker event size V:', lambda p0, p1, f, V: 0 if V < 30 else 30 if V < 100 else 100 if V < 300 else 300 if V < 1000 else 1000)
    grp(' by our fill size:', lambda p0, p1, f, V: 0 if f < 20 else 20 if f < 50 else 50 if f < 100 else 100 if f < 150 else 150)


if __name__ == '__main__':
    main()
