"""PnL = trade edge (vs the filter mean F at each fill) + carry (position x change of F) + residual (F vs mark at the end).
Carry is split by regime, and by the sign of the transient part Y relative to the position.

    uv run python strategies/peter/r3_carry.py ['{"k":v}']
"""
import json, sys
from pathlib import Path
from collections import defaultdict
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1])); sys.path.insert(0, str(HERE))


def main():
    from arena import load_agent
    from simulator.main import MAX_POSITION, MAX_POTENTIAL_POSITION, ORDER_ACCEPT_DELAY_MS, POSITION_BREACH_PENALTY_PER_SECOND
    from simulator.models import OrderStatus
    from simulator.simulator import ExchangeSimulator, SimulationConfig
    over = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    s = load_agent(HERE / 'r3_man.py', over, 'carry')
    st = {'F': None, 'pos': 0}
    edge = defaultdict(float); carry = defaultdict(float)
    onb0, onp0 = s.on_order_book_change, s.on_private_order_update
    kinds = {}
    so0 = s.send_order
    def so(req):
        oid = so0(req)
        kinds[oid] = ('passive' if req.order_type.value == 'limit' else 'ioc', 'turb' if s.turb else 'calm')
        return oid
    s.send_order = so
    def onb(snap, trades):
        onb0(snap, trades)
        f = s.f
        if f is None:
            return
        F = f.mean
        if st['F'] is not None and s.pos:
            reg = 'shock' if (s.sg_t is not None or s.ss_dir) else ('turb' if s.turb else 'calm')
            d = s.pos * (F - st['F'])
            carry[reg] += d
            if reg == 'calm':
                # what moved F: a market taker this event, a market insert, or neither (predict drift / decay)
                src = 'taker' if any(tr.taker_owner == 'market' for tr in trades) else ('our fill' if trades else 'insert/decay')
                carry['calm by ' + src] += d
                if f.Y:
                    carry['calm, Y ' + ('with pos' if f.Y * s.pos > 0 else 'against pos')] += d
        st['F'] = F
    def onp(ev):
        if ev.status in (OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED) and ev.filled_quantity and s.f is not None:
            o = s.orders.get(ev.order_id)
            if o is not None:
                sg = 1 if o[0].name == 'BUY' else -1
                k = kinds.get(ev.order_id, ('?', '?'))
                edge[k] += sg * ev.filled_quantity * (s.f.mean - ev.fill_price)
        onp0(ev)
    s.on_order_book_change, s.on_private_order_update = onb, onp
    cfg = SimulationConfig(max_position=MAX_POSITION, max_potential_position=MAX_POTENTIAL_POSITION,
                           position_breach_penalty_per_second=POSITION_BREACH_PENALTY_PER_SECOND, order_accept_delay_ms=ORDER_ACCEPT_DELAY_MS)
    r = ExchangeSimulator(strategy=s, config=cfg).run(Path('data/round3_public.csv'))
    E = sum(edge.values()); C = sum(v for k, v in carry.items() if k in ('calm', 'turb', 'shock'))
    print(f'pnl {r.pnl/1e3:+.1f}k = edge {E/1e3:+.1f}k + carry {C/1e3:+.1f}k - breach {r.breach_penalty/1e3:.1f}k + residual {(r.pnl - E - C + r.breach_penalty)/1e3:+.1f}k')
    for k, v in sorted(edge.items()):
        print(f'  edge  {k[0]:8} {k[1]:5} {v/1e3:+8.1f}k')
    for k, v in carry.items():
        print(f'  carry {k:14} {v/1e3:+8.1f}k')


if __name__ == '__main__':
    main()
