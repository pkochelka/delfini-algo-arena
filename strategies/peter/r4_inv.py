"""Time-weighted inventory by regime for a strategy on round-4 folds (is the +-100 cap binding in turbulence?).

    uv run python strategies/peter/r4_inv.py [--strategy S.py] [--set '{"k":v}'] [--folds 1,4,7,10]
"""
from __future__ import annotations
import argparse, json, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]


def run(strategy, over, fold):
    sys.path.insert(0, str(ROOT))
    from fastsim.engine import load_strategy, run_backtest
    s = load_strategy(ROOT / strategy, over | {'trade_start_us': fold['trade_start_us']})
    rows, last = [], [None]
    base = s.__class__.on_order_book_change

    def rec(snap, trades):
        base(s, snap, trades)
        t = snap.timestamp_us
        if t >= fold['trade_start_us'] and (last[0] is None or t - last[0] >= 20_000):
            last[0] = t
            rows.append((t, s.pos, s.turb, getattr(s.mo, 'm5', 0.0) if getattr(s, 'mo', None) else 0.0))
    s.on_order_book_change = rec
    r, _ = run_backtest(s, ROOT / fold['path'])
    return r.pnl, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--strategy', default='strategies/peter/r4_trend.py')
    ap.add_argument('--set', default='{}')
    ap.add_argument('--folds', default='1,4,7,10')
    a = ap.parse_args()
    fs = json.loads((ROOT / 'data/splits_r4/wf/folds.json').read_text())['folds']
    fs = [{'path': f'data/splits_r4/wf/{f["file"]}', 'trade_start_us': f['trade_start_us']} for i, f in enumerate(fs, 1)
          if str(i) in a.folds.split(',')]
    with ProcessPoolExecutor(len(fs)) as ex:
        res = list(ex.map(run, [a.strategy] * len(fs), [json.loads(a.set)] * len(fs), fs))
    R = np.array([row for _, rows in res for row in rows], dtype=float)
    pos, turb = np.abs(R[:, 1]), R[:, 2] > 0
    for lab, m in (('calm', ~turb), ('turb', turb)):
        p = pos[m]
        print(f'{lab}: share {m.mean():.2f}  mean|pos| {p.mean():5.1f}  P(|pos|>=90) {(p >= 90).mean():.3f}  '
              f'P(|pos|>=50) {(p >= 50).mean():.3f}  P(flat) {(p == 0).mean():.3f}')
    sg = np.sign(R[:, 1])
    al = sg * R[:, 3]
    m = turb & (pos >= 50)
    print(f'turb |pos|>=50: m5 with the position: mean {al[m].mean():+.2f}, share against by > 2 ticks {(al[m] < -2).mean():.3f}')
    print('pnl', [round(p) for p, _ in res])


if __name__ == '__main__':
    main()
