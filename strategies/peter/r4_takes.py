"""Round 4: does the momentum forecast separate good IOC takes from bad ones? (r4_trend with mo_on, zero weights)

    uv run python strategies/peter/r4_takes.py [--set '{"k":v}'] [--folds 1-12]

Logs every IOC fill of the strategy with the state at send time (regime, F, m1, m5, edge = side * (F - price),
position before), then reports the per-lot markout against the market mid at +1 / 5 / 10 / 30 s, by regime and by
the forecast's sign relative to the trade (with = the forecast points the trade's way), plus whether the take
opened / extended the position or reduced / flipped it.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
H = (1, 5, 10, 30)


def run(over, fold):
    sys.path.insert(0, str(ROOT))
    from fastsim.engine import load_strategy, run_backtest
    from simulator.models import OrderStatus, OrderType
    s = load_strategy(ROOT / 'strategies/peter/r4_trend.py', {'mo_on': 1} | over | {'trade_start_us': fold['trade_start_us']})
    meta, fills, mids = {}, [], []
    send0, onb0, onp0 = s._send, s.on_order_book_change, s.on_private_order_update

    def send(side, price, qty, kind):
        pos = s.pos
        send0(side, price, qty, kind)
        if kind is OrderType.IOC:
            oid = next(reversed(s.orders))
            sg = 1 if side.name == 'BUY' else -1
            snipe = bool(s.ss_dir and s.ss_left > 0) or abs(pos) > 100
            meta[oid] = (sg, s.turb, s.f.mean, s.mo.m1, s.mo.m5, pos, snipe)

    def onb(snap, trades):
        onb0(snap, trades)
        if snap.bids and snap.asks and (not mids or snap.timestamp_us - mids[-1][0] > 50_000):
            mids.append((snap.timestamp_us, (snap.bids[0].price + snap.asks[0].price) / 2))

    def onp(ev):
        if ev.status in (OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED) and ev.order_id in meta:
            sg, turb, F, m1, m5, pos, sn = meta[ev.order_id]
            fills.append((s.now_t * 1e6, sg, ev.filled_quantity, ev.fill_price, turb, sg * (F - ev.fill_price),
                          sg * m1, sg * m5, sg * pos, sn))
        onp0(ev)

    s._send, s.on_order_book_change, s.on_private_order_update = send, onb, onp
    run_backtest(s, ROOT / fold['path'])
    m = np.array(mids)
    out = []
    for f in fills:
        row = list(f)
        for h in H:
            i = min(np.searchsorted(m[:, 0], f[0] + h * 1e6, 'right') - 1, len(m) - 1)
            row.append(f[1] * (m[i, 1] - f[3]))
        out.append(row)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--set', default='{}')
    ap.add_argument('--folds', default='1-12')
    ap.add_argument('--jobs', type=int, default=8)
    a = ap.parse_args()
    lo, hi = map(int, a.folds.split('-'))
    fs = json.loads((ROOT / 'data/splits_r4/wf/folds.json').read_text())['folds'][lo - 1:hi]
    fs = [{'path': f'data/splits_r4/wf/{f["file"]}', 'trade_start_us': f['trade_start_us']} for f in fs]
    with ProcessPoolExecutor(a.jobs) as ex:
        res = list(ex.map(run, [json.loads(a.set)] * len(fs), fs))
    R = np.array([r for rr in res for r in rr], dtype=float)
    q, turb, edge, m1, m5, ppos, sn = R[:, 2], R[:, 4] > 0, R[:, 5], R[:, 6], R[:, 7], R[:, 8], R[:, 9] > 0
    mk = R[:, 10:]
    print(f"{'group':42} {'lots':>7} {'edge':>6} " + ' '.join(f'{f"+{h}s":>7}' for h in H) + '   (per lot)')

    def line(lab, m):
        if m.sum() == 0:
            return
        w = q[m]
        print(f'{lab:42} {w.sum():7.0f} {np.average(edge[m], weights=w):+6.2f} '
              + ' '.join(f'{np.average(mk[m, i], weights=w):+7.2f}' for i in range(len(H))))

    for rl, rm in (('calm', ~turb & ~sn), ('turb', turb & ~sn), ('snipe/xs', sn)):
        line(f'{rl} all', rm)
        if rl == 'snipe/xs':
            continue
        for lab, mm in (('m1 with > 1', m1 > 1), ('m1 with 0..1', (m1 > 0) & (m1 <= 1)), ('m1 against 0..1', (m1 <= 0) & (m1 > -1)),
                        ('m1 against < -1', m1 <= -1)):
            line(f'  {lab}', rm & mm)
        for lab, mm in (('opens/extends (pos with >= 0)', ppos >= 0), ('reduces/flips (pos against)', ppos < 0)):
            line(f'  {lab}', rm & mm)
        for lab, mm in (('edge < 0.5', edge < 0.5), ('edge 0.5..1.5', (edge >= 0.5) & (edge < 1.5)), ('edge 1.5..3', (edge >= 1.5) & (edge < 3)),
                        ('edge >= 3', edge >= 3)):
            line(f'  {lab}', rm & mm)


if __name__ == '__main__':
    main()
