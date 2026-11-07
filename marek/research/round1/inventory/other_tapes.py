"""Robustness: the inventory variants on tapes other than round1 (sample_v2, random_stress), full tape each.

    uv run python research/round1/inventory/other_tapes.py
"""
from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness  # noqa: E402
from harness import ROOT, decompose, run_rec  # noqa: E402

STRAT = str(ROOT / 'strategies' / 'marek' / 'inv_mm.py')
TAPES = {'sample_v2': 5749.896405, 'random_stress': 2005.907913}
VARIANTS = {
    'base': {},
    'lim_add 50': {'lim_add': 50},
    'size_add 50': {'size_add': 50},
    'add_back 1 @80': {'add_back': 1, 'add_back_at': 80},
    'skew_lots 50': {'skew_lots': 50},
    'lim_flip 75': {'lim_flip': 75},
    'unwind 5s': {'unwind_s': 5.0, 'end_us': 'END'},
    # regime switch: inventory control only while our own PnL over the last W s is negative
    'adapt 60s skew50': {'adapt_w_s': 60.0},
    'adapt 120s skew50': {'adapt_w_s': 120.0},
    'adapt 300s skew50': {'adapt_w_s': 300.0},
    'adapt 120s skew25': {'adapt_w_s': 120.0, 'adapt_skew': 25},
    'adapt 120s lim50': {'adapt_w_s': 120.0, 'adapt_skew': 0, 'adapt_lim': 50},
    'adapt 120s skew50+lim50': {'adapt_w_s': 120.0, 'adapt_lim': 50},
}


def job(tape: str, name: str) -> dict:
    harness.set_tape(tape)
    end = TAPES[tape]
    fold = {'name': tape, 'path': str(ROOT / f'data/{tape}.csv'), 'trade_start_us': 0, 'end_s': end}
    ov = {k: (int(end * 1e6) if v == 'END' else v) for k, v in VARIANTS[name].items()}
    r = run_rec(STRAT, ov, fold)
    d = decompose(r, 300)
    return {'tape': tape, 'name': name, **d}


def main() -> None:
    sel = sys.argv[1:] or list(VARIANTS)
    with ProcessPoolExecutor(3) as pool:
        futs = [pool.submit(job, t, n) for t in TAPES for n in sel]
        print(f"{'tape':<14}{'variant':<16}{'pnl':>9}{'spread':>9}{'invPnL':>9}{'slow300':>8}{'liq':>6}{'dmid':>7}"
              f"{'mpos':>6}{'|pos|':>6}{'pin':>5}{'vol':>9}")
        for f in futs:
            d = f.result()
            print(f"{d['tape']:<14}{d['name']:<16}{d['pnl']:>9.0f}{d['spread']:>9.0f}{d['inv_pnl']:>9.0f}{d['slow']:>8.0f}"
                  f"{d['liq']:>6.0f}{d['dmid']:>7.1f}{d['mean_pos']:>6.1f}{d['abs_mean']:>6.1f}{d['pin100']:>5.2f}{d['vol']:>9}",
                  flush=True)


if __name__ == '__main__':
    main()
