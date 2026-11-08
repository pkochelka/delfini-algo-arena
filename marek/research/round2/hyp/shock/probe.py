"""Per-shock PnL probe. uv run python marek/research/round2/hyp/shock/probe.py STRAT.py [--set k=v] [--tapes r2,s1,..]
Per shock: pnl from onset-0.5 to onset+8 (marked at filter F), sniper fired/dir/lots, pos at onset+3, |jump|."""
import argparse, ast, json, sys, importlib.util, pathlib, csv
from concurrent.futures import ProcessPoolExecutor
ROOT = pathlib.Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
TAPES = {'r2': 'data/round2.csv', **{f's{i}': f'data/synth/r2_v2_seed{i}.csv' for i in range(1, 8)}}

def onsets(tape):
    if tape == 'r2':
        rows = list(csv.DictReader(open(ROOT / 'marek/research/round2/flow/shock_events.csv')))
        return [float(r['onset']) for r in rows], [None] * len(rows)
    d = json.load(open(ROOT / f'data/synth/r2_v2_seed{tape[1:]}.truth.json'))
    return d['onsets'], d['jumps']

def run(args):
    strat, over, tape, oracle = args
    spec = importlib.util.spec_from_file_location('stratmod', strat)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    m.PARAMS.update(over)
    from simulator.main import (MAX_POSITION, MAX_POTENTIAL_POSITION, ORDER_ACCEPT_DELAY_MS,
                                POSITION_BREACH_PENALTY_PER_SECOND)
    from simulator.models import OrderStatus
    from simulator.simulator import ExchangeSimulator, SimulationConfig
    base = m.FVAShock
    o_on, o_J = onsets(tape)
    if oracle:
        m.PARAMS.update({'ss_w0': 99.0, 'ss_w1': 99.0, 'ss_t0': 99.0, 'ss_t1': 99.0})
    class P(base):
        def __init__(s):
            super().__init__(); s.oi = 0; s.fills = []; s.path = []; s.nxt = 0.0; s.fires = []
        def on_private_order_update(s, e):
            o = s.orders.get(e.order_id)
            if o is not None and e.status in (OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED):
                s.fills.append((e.timestamp_us / 1e6, 1 if o[0] is m.BUY else -1, e.filled_quantity, e.fill_price))
            super().on_private_order_update(e)
        def on_order_book_change(s, snap, tr):
            n = s.n_snipes
            super().on_order_book_change(snap, tr)
            if oracle and s.oi < len(o_on) and s.now_t >= o_on[s.oi] + 3.0:
                J = o_J[s.oi]; s.oi += 1
                if J is not None and abs(J) >= 3 and snap.bids and snap.asks and s.ss_dir == 0:
                    d = 1 if J > 0 else -1
                    s.ss_dir = d; s.ss_ref = snap.asks[0].price if d > 0 else snap.bids[0].price
                    lim = int(abs(J)) - 2 if oracle == 1 else s.p['ss_lim']
                    s.p['ss_lim'] = lim
                    s.ss_left = s.p['ss_shots']; s.ss_next = s.now_t; s.ss_hold_until = s.now_t + s.p['ss_hold']
                    s.n_snipes += 1
                    side = m.BUY if d > 0 else m.SELL
                    q = s._room_side(side); px = s.ss_ref + lim * d
                    if q > 0 and px > 0: s._send(side, px, q, m.OrderType.IOC)
                    s.ss_left -= 1; s.ss_next = s.now_t + s.p['ss_gap_ms'] * 1e-3
            if s.n_snipes != n: s.fires.append((s.now_t, s.ss_dir))
            if s.f is not None and s.now_t >= s.nxt:
                s.path.append((s.now_t, s.f.mean, s.pos)); s.nxt = s.now_t + 0.01
    st = P()
    cfg = SimulationConfig(max_position=MAX_POSITION, max_potential_position=MAX_POTENTIAL_POSITION,
                           position_breach_penalty_per_second=POSITION_BREACH_PENALTY_PER_SECOND,
                           order_accept_delay_ms=ORDER_ACCEPT_DELAY_MS)
    r = ExchangeSimulator(strategy=st, config=cfg).run(ROOT / TAPES[tape])
    import numpy as np, bisect
    pt = np.array(st.path); fl = st.fills; ft = [f[0] for f in fl]
    def val(t):  # cash + pos*F at t
        i = bisect.bisect_right(ft, t)
        cash = -sum(f[1] * f[2] * f[3] for f in fl[:i]); pos = sum(f[1] * f[2] for f in fl[:i])
        j = max(np.searchsorted(pt[:, 0], t, side='right') - 1, 0)
        return cash + pos * pt[j, 1], pos, pt[j, 1]
    ons, jumps = onsets(tape); out = []
    for o, J in zip(ons, jumps):
        v0, p0, F0 = val(o - 0.5); v3, p3, _ = val(o + 2.97); v35, p35, _ = val(o + 3.5); v8, p8, F8 = val(o + 8)
        fire = [d for t, d in st.fires if o < t < o + 4]
        lots = sum(f[2] for f in fl if o + 2.95 < f[0] < o + 3.35)
        out.append(dict(on=round(o, 2), J=round(J, 1) if J is not None else round(F8 - F0, 1), pre=round(v3 - v0),
                        jmp=round(v35 - v3), post=round(v8 - v35), tot=round(v8 - v0), p0=p0, p3=p3, p35=p35, p8=p8,
                        fire=fire[0] if fire else 0, lots=lots))
    return tape, r.pnl - r.breach_penalty, out

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('strategy'); ap.add_argument('--set', nargs='*', default=[])
    ap.add_argument('--tapes', default='r2,s1,s2'); ap.add_argument('--out', default=''); ap.add_argument('--oracle', type=int, default=0)
    ap.add_argument('-q', action='store_true')
    a = ap.parse_args()
    over = {k: ast.literal_eval(v) for k, v in (x.split('=', 1) for x in a.set)}
    tp = a.tapes.split(',')
    allr = {}
    with ProcessPoolExecutor(len(tp)) as ex:
        for tape, pnl, out in ex.map(run, [(a.strategy, over, t, a.oracle) for t in tp]):
            allr[tape] = out
            print(f'== {tape} pnl {pnl:.0f}  shocks tot {sum(x["tot"] for x in out)}  jmp {sum(x["jmp"] for x in out)}  '
                  f'fired {sum(1 for x in out if x["fire"])}/{len(out)}  sum|J|*100 {sum(abs(x["J"]) for x in out)*100:.0f}')
            if not a.q:
                for x in out: print('  ', x)
    if a.out: json.dump(allr, open(a.out, 'w'))
