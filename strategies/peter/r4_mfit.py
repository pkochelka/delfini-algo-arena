"""Fit r4_trend's momentum forecast (class Mom) on the strategy's own live features.

    uv run python strategies/peter/r4_mfit.py data/round4_public.csv [--split 4500] [--strategy strategies/peter/r4_trend.py]

Runs the strategy without trading (mo_on 1, zero coefficients), samples Mom's feature vector z, the filter mean F and
P(volatile) every 0.1 s, and regresses F(t+h) - F(t) on z (h = 1, 5 s) separately for calm (pv < 0.5) and volatile
samples, on t < split; reports out-of-sample R2 on t >= split and prints the PARAMS line fitted on the whole tape.
Samples whose target is beyond ymax (shock jumps) are dropped from the fit.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from fastsim.engine import load_strategy, run_backtest  # noqa: E402


def record(strategy: str, tape: str, dt: float = 0.1):
    s = load_strategy(ROOT / strategy, {'trade_start_us': 10**15, 'mo_on': 1})
    rows = []
    nxt = [0.0]
    base = s.__class__.on_order_book_change

    def rec(snapshot, trades):
        base(s, snapshot, trades)
        t = snapshot.timestamp_us * 1e-6
        if s.mo is not None and s.mo.t is not None and t >= nxt[0]:
            nxt[0] = t + dt
            rows.append([t, s.f.mean, s.f.pv, s.sg_t is not None] + list(s.mo.z[:6]))

    s.on_order_book_change = rec
    run_backtest(s, ROOT / tape)
    return np.array(rows)


def fit(A, y):
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return beta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('tape')
    ap.add_argument('--strategy', default='strategies/peter/r4_trend.py')
    ap.add_argument('--split', type=float, default=4500.0)
    ap.add_argument('--ymax', type=float, default=30.0)
    ap.add_argument('--cache')
    a = ap.parse_args()
    cache = Path(a.cache or ROOT / 'logs' / 'r4' / f'mfeat_{Path(a.tape).stem}.npy')
    if cache.exists():
        R = np.load(cache)
    else:
        R = record(a.strategy, a.tape)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, R)
    t, F, pv, shk = R[:, 0], R[:, 1], R[:, 2], R[:, 3] > 0
    Z = np.column_stack([R[:, 4:10], np.ones(len(R))])
    out = {}
    for h in (1, 5):
        k = np.searchsorted(t, t + h)
        ok = k < len(t)
        y = np.full(len(t), np.nan)
        y[ok] = F[k[ok]] - F[ok]
        good = ok & np.isfinite(y) & (np.abs(y) < a.ymax) & ~shk
        for r, m in (('c', pv < 0.5), ('v', pv >= 0.5)):
            tr = good & m & (t < a.split)
            te = good & m & (t >= a.split)
            b = fit(Z[tr], y[tr])
            e = y[te] - Z[te] @ b
            r2 = 1 - e @ e / ((y[te] - y[te].mean()) @ (y[te] - y[te].mean()))
            ball = fit(Z[good & m], y[good & m])
            out[f'mo_b{h}{r}'] = [round(float(v), 4) for v in ball]
            print(f'h {h} {r}: n {tr.sum():6d}/{te.sum():6d}  oos R2 {r2:+.3f}  train-fit {np.round(b, 3).tolist()}')
    print('PARAMS.update(' + repr(out) + ')')


if __name__ == '__main__':
    main()
