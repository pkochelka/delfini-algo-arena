"""A/B inventory variants (strategies/marek/inv_mm.py) on the round1 walk-forward folds wf01-10.

    uv run python research/round1/inventory/sweep_inv.py [variant ...]

Per variant: total PnL, delta vs baseline, per-fold-delta sd (paired), fold sd, PnL split into spread capture /
inventory PnL / slow (300 s) directional part, and the PnL on the folds where mid rose vs fell.
"""
from __future__ import annotations

import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import ROOT, decompose, run_rec  # noqa: E402
from tune import load_folds  # noqa: E402

STRAT = str(ROOT / 'strategies' / 'marek' / 'inv_mm.py')
FOLDS = load_folds(ROOT / 'data/splits_round1/wf', list(range(1, 11)))
END = 'END'  # placeholder: replaced by each fold's end time (µs)

VARIANTS: dict[str, dict] = {
    'base': {},
    # asymmetric caps: limit the adding side
    'lim_add 75': {'lim_add': 75},
    'lim_add 50': {'lim_add': 50},
    'lim_add 25': {'lim_add': 25},
    # limit how far the reducing side flips us through flat
    'lim_flip 75': {'lim_flip': 75},
    'lim_flip 50': {'lim_flip': 50},
    'lim_flip 0': {'lim_flip': 0},
    # size on the adding side
    'size_add 50': {'size_add': 50},
    'size_add 25': {'size_add': 25},
    # adding side behind the touch once loaded
    'add_back 1 @50': {'add_back': 1, 'add_back_at': 50},
    'add_back 1 @80': {'add_back': 1, 'add_back_at': 80},
    # ladder_mm switches tried on sample_v2
    'skew_lots 50': {'skew_lots': 50},
    'skew_lots 25': {'skew_lots': 25},
    'improve_reduce': {'improve_reduce': 1},
    'flat_at 100 to 50': {'flat_at': 100, 'flat_to': 50},
    'flat_at 100 to 0': {'flat_at': 100, 'flat_to': 0},
    # end of tape
    'unwind 5s': {'end_us': END, 'unwind_s': 5.0},
    'unwind 30s': {'end_us': END, 'unwind_s': 30.0},
    'unwind 120s': {'end_us': END, 'unwind_s': 120.0},
    'unwind 30s + ioc 2s': {'end_us': END, 'unwind_s': 30.0, 'unwind_ioc_s': 2.0},
    # regime switch: inventory control only while our own PnL over the last W s is negative
    'adapt 60s skew50': {'adapt_w_s': 60.0},
    'adapt 120s skew50': {'adapt_w_s': 120.0},
    'adapt 300s skew50': {'adapt_w_s': 300.0},
    'adapt 120s skew25': {'adapt_w_s': 120.0, 'adapt_skew': 25},
    'adapt 120s lim50': {'adapt_w_s': 120.0, 'adapt_skew': 0, 'adapt_lim': 50},
    'adapt 120s skew50+lim50': {'adapt_w_s': 120.0, 'adapt_lim': 50},
}


def job(name: str, fold: dict) -> dict:
    ov = {k: (int(round(fold['end_s'] * 1e6)) if v == END else v) for k, v in VARIANTS[name].items()}
    r = run_rec(STRAT, ov, fold)
    d = decompose(r, 300)
    return {'fold': r['fold'], 'pnl': r['pnl'], 'q95': r['q95'], 'breach': r['breach'], **{
        k: d[k] for k in ('spread', 'inv_pnl', 'slow', 'liq', 'dmid', 'abs_mean', 'vol', 'pin100')}}


def main() -> None:
    sel = sys.argv[1:] or list(VARIANTS)
    if 'base' not in sel:
        sel = ['base', *sel]
    with ProcessPoolExecutor(3) as pool:
        jobs = {n: [pool.submit(job, n, f) for f in FOLDS] for n in sel}
        base = None
        print(f"{'variant':<22}{'total':>9}{'delta':>8}{'d_sd':>6}{'d_min':>7}{'d_max':>7}{'foldsd':>7}{'spread':>9}"
              f"{'invPnL':>9}{'slow':>7}{'liq':>6}{'up5':>8}{'down5':>8}{'|pos|':>6}{'pin':>5}{'vol':>9}{'brch':>6}")
        for n, futs in jobs.items():
            res = [f.result() for f in futs]
            pn = [r['pnl'] for r in res]
            if base is None:
                base = pn
            dl = [a - b for a, b in zip(pn, base)]
            up = sum(r['pnl'] for r in res if r['dmid'] > 0)
            dn = sum(r['pnl'] for r in res if r['dmid'] <= 0)
            S = lambda k: sum(r[k] for r in res)  # noqa: E731
            print(f"{n:<22}{sum(pn):>9.0f}{sum(dl):>8.0f}{statistics.pstdev(dl):>6.0f}{min(dl):>7.0f}{max(dl):>7.0f}"
                  f"{statistics.pstdev(pn):>7.0f}{S('spread'):>9.0f}{S('inv_pnl'):>9.0f}{S('slow'):>7.0f}{S('liq'):>6.0f}"
                  f"{up:>8.0f}{dn:>8.0f}{S('abs_mean') / 10:>6.1f}{S('pin100') / 10:>5.2f}{S('vol'):>9}{S('breach'):>6.0f}",
                  flush=True)


if __name__ == '__main__':
    main()
