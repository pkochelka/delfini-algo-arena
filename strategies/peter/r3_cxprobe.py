"""Why does r3_man cancel its passive orders? Re-evaluates the keep rule for every live passive order at each _quote and
tags each cancel with the first failing condition; also order lifetimes and fills.

    uv run python strategies/peter/r3_cxprobe.py ['{"k":v}']
"""
import json, math, sys
from collections import Counter
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1])); sys.path.insert(0, str(HERE))


def main():
    from arena import load_agent
    from simulator.main import MAX_POSITION, MAX_POTENTIAL_POSITION, ORDER_ACCEPT_DELAY_MS, POSITION_BREACH_PENALTY_PER_SECOND
    from simulator.models import OrderType
    from simulator.simulator import ExchangeSimulator, SimulationConfig
    over = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    s = load_agent(HERE / 'r3_man.py', over, 'cx')
    born, why, life = {}, Counter(), []
    so0, cx0, q0 = s.send_order, s.cancel_order, s._quote
    ctx = {}
    def so(req):
        oid = so0(req)
        if req.order_type is OrderType.LIMIT:
            born[oid] = s.now_t
        return oid
    def cx(oid):
        if oid in born:
            o = s.orders.get(oid)
            c = ctx
            r = 'other'
            if o is not None and c:
                px = o[1]
                if c['turb']:
                    r = 'turb/shock/snipe'
                elif o[0].name == 'BUY':
                    F = c['F']
                    r = 'edge<keep' if F - px < c['kp'] else 'too deep' if px < c['tb'] - c['kd'] else 'crossed' if px >= c['ba'] else 'unreach' if px < round(F - c['ro']) else 'other'
                else:
                    F = c['F']
                    r = 'edge<keep' if px - F < c['kp'] else 'too deep' if px > c['ta'] + c['kd'] else 'crossed' if px <= c['bb'] else 'unreach' if px > round(F + c['ro']) else 'other'
            why[r] += 1
            life.append(s.now_t - born.pop(oid))
        cx0(oid)
    def quote(f, bb, ba, bids, asks):
        p = s.p
        F = f.mean - p['y_k'] * f.Y
        turb = s._turbulent(f) or s.sg_t is not None or bool(s.ss_dir)
        ctx.update(F=F, turb=turb, kp=p['theta_keep'], kd=p['keep_depth'], ro=p['reach_off'], bb=bb, ba=ba,
                   tb=min(math.floor(F - p['theta']), ba - 1), ta=max(math.ceil(F + p['theta']), bb + 1))
        q0(f, bb, ba, bids, asks)
    s.send_order, s.cancel_order, s._quote = so, cx, quote
    cfg = SimulationConfig(max_position=MAX_POSITION, max_potential_position=MAX_POTENTIAL_POSITION,
                           position_breach_penalty_per_second=POSITION_BREACH_PENALTY_PER_SECOND, order_accept_delay_ms=ORDER_ACCEPT_DELAY_MS)
    r = ExchangeSimulator(strategy=s, config=cfg).run(Path('data/round3_public.csv'))
    L = np.array(life)
    print(f'pnl {r.pnl:+.0f}  passive orders sent {len(born) + len(L)}  cancelled {len(L)}  ({len(L)/6000:.1f}/s)')
    for k, v in why.most_common():
        print(f'  {k:18} {v:8d} ({v/len(L):.2f})')
    print('  lifetime of cancelled orders: p10 %.3f p50 %.3f p90 %.3f s' % tuple(np.percentile(L, [10, 50, 90])))


if __name__ == '__main__':
    main()
