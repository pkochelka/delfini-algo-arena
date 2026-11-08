"""How much take edge does the +-100 cap block? Wraps r3_man's take rule: at every take signal, the edge-weighted market
volume through the threshold (vs the filter mean F) that was taken vs blocked by the room, split by regime.

    uv run python strategies/peter/r3_capprobe.py [data/round3_public.csv] ['{"k":v}']
"""
import json, math, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1])); sys.path.insert(0, str(HERE))


def main():
    from arena import load_agent
    from simulator.main import MAX_POSITION, MAX_POTENTIAL_POSITION, ORDER_ACCEPT_DELAY_MS, POSITION_BREACH_PENALTY_PER_SECOND
    from simulator.simulator import ExchangeSimulator, SimulationConfig
    tape = sys.argv[1] if len(sys.argv) > 1 else 'data/round3_public.csv'
    over = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    s = load_agent(HERE / 'r3_man.py', over, 'cap')
    st = {}
    q0 = s._quote
    seen = set()
    hist = {}

    def quote(f, bb, ba, bids, asks):
        p = s.p
        F, sd = f.mean - p['y_k'] * f.Y, f.sd
        turb = s.turb
        take, tsd = (p['take_t'], p['take_sd_t']) if turb else (p['take'], p['take_sd'])
        tk = take + tsd * sd
        pos = s.pos
        rest_b = sum(o[2] for o in s.orders.values() if o[0].name == 'BUY')
        rest_a = sum(o[2] for o in s.orders.values() if o[0].name == 'SELL')
        reg = 'turb' if turb else 'calm'
        for side, views, lim_ok, room in (('B', asks, lambda px: px <= F - tk, 100 - pos - rest_b),
                                          ('S', bids, lambda px: px >= F + tk, 100 + pos - rest_a)):
            avail = []
            for o in views:
                if not lim_ok(o.price):
                    break
                if o.owner == 'market' and o.order_id not in seen:
                    avail.append((o.order_id, o.quantity, abs(F - o.price)))
            if not avail:
                continue
            r = max(room, 0)
            for oid, q, e in avail:  # each resting order counted once (first time it is through the threshold)
                seen.add(oid)
                tq = min(q, r); r -= tq
                d = st.setdefault(reg, [0, 0.0, 0, 0.0])
                d[0] += tq; d[1] += tq * e; d[2] += q - tq; d[3] += (q - tq) * e
                b = 0 if e < 2 else 1 if e < 4 else 2 if e < 8 else 3
                h = hist.setdefault(reg, [[0, 0.0] for _ in range(4)])
                h[b][0] += q - tq; h[b][1] += (q - tq) * e
        q0(f, bb, ba, bids, asks)
    s._quote = quote
    cfg = SimulationConfig(max_position=MAX_POSITION, max_potential_position=MAX_POTENTIAL_POSITION,
                           position_breach_penalty_per_second=POSITION_BREACH_PENALTY_PER_SECOND, order_accept_delay_ms=ORDER_ACCEPT_DELAY_MS)
    r = ExchangeSimulator(strategy=s, config=cfg).run(Path(tape))
    print(f'pnl {r.pnl:+.0f}')
    for reg, (tq, te, bq, be) in st.items():
        print('   blocked by edge [<2, 2-4, 4-8, 8+]: ' + ' | '.join(f'{q:8d} lots {v/1e3:7.1f}k' for q, v in hist.get(reg, [])))
        print(f'{reg}: taken {tq:8d} lots edge {te/1e3:7.1f}k ({te/max(tq,1):.2f}/lot) | blocked by room {bq:8d} lots edge {be/1e3:7.1f}k ({be/max(bq,1):.2f}/lot)')


if __name__ == '__main__':
    main()
