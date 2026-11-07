"""Inventory pinning analysis of ladder_mm on the round1 walk-forward folds.

    uv run python research/round1/inventory/analyze_baseline.py [strategy.py] [json overrides]

Prints per-fold PnL decomposition (spread capture / inventory PnL split slow vs fast / liquidation),
pinning stats and episode durations, and the markout of the flow we miss on the blocked side.
"""
from __future__ import annotations

import json
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import ROOT, decompose, mid_at, quotes, run_rec  # noqa: E402
from tune import load_folds  # noqa: E402

FOLDS = load_folds(ROOT / 'data/splits_round1/wf', list(range(1, 11)))


def episodes(r: dict, thr: int = 95):
    """Durations (s) of maximal intervals with |pos| >= thr, by sign."""
    t, q = r['f_t'], r['f_q']
    pos = np.cumsum(q)
    ends = np.append(t[1:], r['t1_us'])
    out = {1: [], -1: []}
    cur, start = 0, None
    for p, a, b in zip(pos, t, ends):
        s = 1 if p >= thr else -1 if p <= -thr else 0
        if s != cur:
            if cur and start is not None:
                out[cur].append((a - start) / 1e6)
            cur, start = s, a if s else None
    if cur and start is not None:
        out[cur].append((r['t1_us'] - start) / 1e6)
    return out


def blocked_flow(results):
    """Markout of the tape's aggressive orders that hit the side we can't quote (pinned) vs the rest."""
    from analysis import load_tape
    from analysis.features import taker_orders
    tape = load_tape(ROOT / 'data/round1.csv', verbose=False)
    tk = taker_orders(tape)
    t_us = tk['t_us'].to_numpy()
    sign = tk['sign'].to_numpy()
    qty = np.minimum(tk['qty'].to_numpy(), 100)
    tq, mid, bid, ask = quotes()
    i0 = np.clip(np.searchsorted(tq, t_us, side='left') - 1, 0, len(tq) - 1)
    # passive counterpart price: we'd buy at the bid when a seller hits it, sell at the ask for a buyer
    touch = np.where(sign < 0, bid[i0], ask[i0])
    # our position just before each taker order
    posb = np.zeros(len(t_us))
    inwin = np.zeros(len(t_us), bool)
    for r in results:
        m = (t_us >= r['t0_us']) & (t_us < r['t1_us'])
        inwin |= m
        pos = np.concatenate([[0], np.cumsum(r['f_q'])])
        posb[m] = pos[np.searchsorted(r['f_t'], t_us[m], side='left')]
    blocked = ((sign < 0) & (posb >= 95)) | ((sign > 0) & (posb <= -95))
    adding_open = ((sign < 0) & (posb < 95) & (posb > 0)) | ((sign > 0) & (posb > -95) & (posb < 0))
    reducing = ((sign < 0) & (posb < 0)) | ((sign > 0) & (posb > 0))
    print('\nmaker markout of the flow (ticks, our direction, weight min(qty,100)), mid(t+h) - touch:')
    print(f"{'group':<28}{'orders':>8}{'lots':>10}" + ''.join(f'{h:>8}' for h in ('0', '1s', '5s', '30s', '120s')))
    for name, m in (('blocked side (pinned)', blocked & inwin), ('adding side, not pinned', adding_open & inwin),
                    ('reducing side', reducing & inwin), ('all', inwin)):
        w = qty[m]
        row = f'{name:<28}{m.sum():>8}{w.sum():>10}'
        for h in (0, 1, 5, 30, 120):
            mo = -sign[m] * (mid_at(t_us[m] + int(h * 1e6)) - touch[m])
            row += f'{np.sum(mo * w) / w.sum():>8.3f}'
        print(row)
    # blocked-side lots per second of pinned time
    return blocked & inwin


def main() -> None:
    strat = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / 'strategies/peter/ladder_mm.py')
    ov = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    with ProcessPoolExecutor(3) as pool:
        results = list(pool.map(run_rec, [strat] * 10, [ov] * 10, FOLDS))
    print(f"{'fold':<5}{'pnl':>8}{'spread':>9}{'invPnL':>9}{'slow10':>8}{'slow60':>8}{'slow300':>8}{'liq':>6}"
          f"{'dmid':>7}{'mpos':>7}{'|pos|':>6}{'pin100':>7}{'L95':>6}{'S95':>6}{'vol':>8}")
    tot = {}
    for r in results:
        d = {s: decompose(r, s) for s in (10, 60, 300)}
        d0 = d[60]
        print(f"{r['fold']:<5}{d0['pnl']:>8.0f}{d0['spread']:>9.0f}{d0['inv_pnl']:>9.0f}{d[10]['slow']:>8.0f}"
              f"{d[60]['slow']:>8.0f}{d[300]['slow']:>8.0f}{d0['liq']:>6.0f}{d0['dmid']:>7.1f}{d0['mean_pos']:>7.1f}"
              f"{d0['abs_mean']:>6.1f}{d0['pin100']:>7.2f}{d0['long95']:>6.2f}{d0['short95']:>6.2f}{d0['vol']:>8}")
        for k, v in (('pnl', d0['pnl']), ('spread', d0['spread']), ('inv', d0['inv_pnl']), ('s10', d[10]['slow']),
                     ('s60', d[60]['slow']), ('s300', d[300]['slow']), ('liq', d0['liq']), ('whole', d0['whole_drift'])):
            tot[k] = tot.get(k, 0) + v
    pn = [r['pnl'] for r in results]
    print('TOTAL', {k: round(v) for k, v in tot.items()}, 'fold sd', round(statistics.pstdev(pn)))
    ep = {1: [], -1: []}
    for r in results:
        e = episodes(r)
        ep[1] += e[1]
        ep[-1] += e[-1]
    for s, name in ((1, 'long >=95'), (-1, 'short <=-95')):
        a = np.array(ep[s])
        print(f'{name}: {len(a)} episodes, total {a.sum():.0f} s, median {np.median(a):.2f} s, '
              f'mean {a.mean():.2f} s, p90 {np.percentile(a, 90):.1f} s, max {a.max():.0f} s')
    # fills by our position before the fill: adding vs reducing
    add = red = 0
    for r in results:
        pos_b = np.cumsum(r['f_q']) - r['f_q']
        a_ = np.sign(r['f_q']) == np.sign(pos_b)
        add += np.abs(r['f_q'][a_]).sum()
        red += np.abs(r['f_q'][~a_]).sum()
    print(f'lots filled adding to |pos| (or from flat): {add}, reducing/flipping: {red}')
    np.savez_compressed(Path('/private/tmp') / 'inv_baseline_fills.npz', **{
        f"{r['fold']}_{k}": r[k] for r in results for k in ('f_t', 'f_q', 'f_px')})
    if not ov:
        blocked_flow(results)


if __name__ == '__main__':
    main()
