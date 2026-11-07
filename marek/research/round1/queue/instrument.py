"""Instrumented replay: where in the queue do our orders sit, and what happens at touch transitions.

    uv run python research/round1/queue/instrument.py --strategy strategies/peter/ladder_mm.py \
        --params '{}' --folds 1-10 --out research/round1/queue/out/base.pkl

Wraps a strategy instance (no change to the strategy file) and, after every public snapshot, reads
the per-order book (best price first, FIFO within a level) to record:

- orders: per own order its send / activation time, price relative to the market touch at activation
  (0 = at touch, -1 = one tick behind, +1 = inside), market lots ahead at activation and at its last
  snapshot, filled lots, end time.
- fills: every own fill with the market lots that were ahead of the order on the snapshot before it.
- trans: every change of the market touch price (own orders excluded) on a side: direction
  (+1 = the side improved, -1 = it worsened / the level broke), our resting lots already at the new
  price when it formed, our room on that side, and when (and how deep in the queue) we first had an
  active order there.

Fold timestamps are absolute tape times, so markouts can be taken from analysis.load_tape(round1).
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'strategies' / 'peter'))


def _scan(orders, ours: dict, now: int, recs: dict, side_tag: str):
    """Market touch price/qty and, for each own order seen, market lots ahead of it at its level."""
    mt = None
    mq = 0
    level_px = None
    ahead = 0
    nlev = 0
    for o in orders:
        if o.price != level_px:
            level_px = o.price
            ahead = 0
            nlev += 1
            if nlev > 3:
                break
        if o.owner == 'market':
            if mt is None:
                mt = o.price
            if o.price == mt:
                mq += o.quantity
            ahead += o.quantity
        else:
            r = recs.get(o.order_id)
            if r is not None:
                if r['act_t'] is None:
                    r['act_t'] = now
                    r['ahead_act'] = ahead
                    r['_pending_rel'] = True
                r['ahead_last'] = ahead
                ours[o.order_id] = o.price
    return mt, mq


def run(strategy: str, overrides: dict, fold: dict) -> dict:
    from arena import load_agent
    from simulator.main import (MAX_POSITION, MAX_POTENTIAL_POSITION, ORDER_ACCEPT_DELAY_MS,
                                POSITION_BREACH_PENALTY_PER_SECOND)
    from simulator.models import OrderStatus, Side
    from simulator.simulator import ExchangeSimulator, SimulationConfig

    strat = load_agent(Path(strategy), overrides | {'trade_start_us': fold['trade_start_us']}, 'instr')
    start = fold['trade_start_us']
    recs: dict[str, dict] = {}
    fills: list[tuple] = []
    trans: list[dict] = []
    pending: dict[str, dict | None] = {'B': None, 'A': None}
    last_mt: dict[str, int | None] = {'B': None, 'A': None}
    state = {'pos': 0}

    orig_send = strat.send_order
    orig_book = strat.on_order_book_change
    orig_priv = strat.on_private_order_update

    def send_order(req):
        oid = orig_send(req)
        recs[oid] = {'side': req.side.value, 'price': req.price, 'qty': req.quantity, 'type': req.order_type.value,
                     'sent_t': strat.now_us(), 'act_t': None, 'ahead_act': None, 'ahead_last': None,
                     'rel_act': None, 'filled': 0, 'end_t': None, 'end': None, 'touch_at_send': None}
        return oid

    def on_priv(ev):
        r = recs.get(ev.order_id)
        if r is not None:
            st = ev.status
            if st in (OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED):
                q = ev.filled_quantity
                r['filled'] += q
                state['pos'] += q if r['side'] == 'B' else -q
                fills.append((ev.timestamp_us, ev.order_id, r['side'], ev.fill_price, q, r['ahead_last']))
            if st in (OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED):
                r['end_t'] = ev.timestamp_us
                r['end'] = st.value
        orig_priv(ev)

    def on_book(snap, trades):
        now = snap.timestamp_us
        if now >= start:
            for side_tag, orders in (('B', snap.bids), ('A', snap.asks)):
                ours: dict[str, int] = {}
                mt, mq = _scan(orders, ours, now, recs, side_tag)
                if mt is None:
                    continue
                sgn = 1 if side_tag == 'B' else -1
                for oid in ours:
                    r = recs[oid]
                    if r.pop('_pending_rel', False):
                        r['rel_act'] = (r['price'] - mt) * sgn
                        r['mq_act'] = mq
                old = last_mt[side_tag]
                if old is not None and mt != old:
                    traded_old = sum(t.quantity for t in trades if t.price == old and
                                     t.book_side == ('bid' if side_tag == 'B' else 'ask'))
                    have = sum(1 for oid, px in ours.items() if px == mt)
                    have_q = sum(recs[oid]['qty'] - recs[oid]['filled'] for oid, px in ours.items() if px == mt)
                    p = strat.p
                    rest = sum(o[2] for oid, o in strat.orders.items() if o[0].value == side_tag)
                    room = (p['cap'] - (state['pos'] + rest)) if side_tag == 'B' else (p['cap'] + state['pos'] - rest)
                    rec = {'t': now, 'side': side_tag, 'old': old, 'new': mt, 'dir': (mt - old) * sgn,
                           'traded_old': traded_old, 'mq_new': mq, 'have_q': have_q, 'pos': state['pos'],
                           'room': room, 'join_t': None, 'join_ahead': None,
                           'had_old': any(px == old for px in ours.values())}
                    if have:
                        rec['join_t'] = now
                        first = min((oid for oid, px in ours.items() if px == mt), key=lambda x: recs[x]['act_t'])
                        rec['join_ahead'] = recs[first]['ahead_last']
                    else:
                        pending[side_tag] = rec
                    trans.append(rec)
                elif pending[side_tag] is not None:
                    pr = pending[side_tag]
                    at = [oid for oid, px in ours.items() if px == mt]
                    if at:
                        first = min(at, key=lambda x: recs[x]['act_t'])
                        pr['join_t'] = now
                        pr['join_ahead'] = recs[first]['ahead_last']
                        pending[side_tag] = None
                if old is None or mt != old:
                    if old is not None and pending[side_tag] is not None and pending[side_tag]['t'] != now:
                        pending[side_tag] = None
                    last_mt[side_tag] = mt
        orig_book(snap, trades)

    strat.send_order = send_order
    strat.on_order_book_change = on_book
    strat.on_private_order_update = on_priv
    cfg = SimulationConfig(max_position=MAX_POSITION, max_potential_position=MAX_POTENTIAL_POSITION,
                           position_breach_penalty_per_second=POSITION_BREACH_PENALTY_PER_SECOND,
                           order_accept_delay_ms=ORDER_ACCEPT_DELAY_MS)
    res = ExchangeSimulator(strategy=strat, config=cfg).run(Path(fold['path']))
    return {'fold': fold['name'], 'pnl': res.pnl, 'orders': recs, 'fills': fills, 'trans': trans}


def main() -> None:
    from tune import load_folds
    ap = argparse.ArgumentParser()
    ap.add_argument('--strategy', default=str(ROOT / 'strategies/peter/ladder_mm.py'))
    ap.add_argument('--params', default='{}')
    ap.add_argument('--folds', default='1-10')
    ap.add_argument('--out', required=True)
    ap.add_argument('--jobs', type=int, default=3)
    a = ap.parse_args()
    lo, _, hi = a.folds.partition('-')
    folds = load_folds(ROOT / 'data/splits_round1/wf', list(range(int(lo), int(hi or lo) + 1)))
    with ProcessPoolExecutor(max_workers=a.jobs) as pool:
        out = list(pool.map(run, [a.strategy] * len(folds), [json.loads(a.params)] * len(folds), folds))
    for r in out:
        print(r['fold'], r['pnl'], len(r['orders']), len(r['fills']), len(r['trans']))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, 'wb') as fh:
        pickle.dump(out, fh)


if __name__ == '__main__':
    main()
