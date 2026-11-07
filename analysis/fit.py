"""Offline parameter fits for strategies, from a replayed tape.

    uv run python -m analysis.fit data/round1.csv [--window 0 1800]

prints the parameters and writes them to data/.cache/<stem>-params.json.
Submissions can't read files, so copy the numbers into the strategy.

- sigma: realised mid volatility, ticks per sqrt-second, at 100 ms (the
  sampling of `strategies/avellaneda_stoikov.py`) and 1 s.
- k_depth: the strategy's own estimator, k = ln(1 + 1 / mean sweep depth past the touch).
- A, k_intensity: least-squares fit of log lambda(delta) = log A - k * delta, where
  lambda(delta) is the per-side rate of aggressive orders reaching delta ticks
  from the mid (`features.fill_intensity`).
- OFI beta, the imbalance -> 1 s mid move table, and taker markouts.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from analysis import features as F
from analysis.cache import cache_path, load_tape
from analysis.replay import Tape


def fit_intensity(tape: Tape, window: F.Window = None) -> dict[str, float]:
    """Fit lambda(delta) = A * exp(-k * delta) on the deltas from the typical half spread out to where fills run out."""
    lam = F.fill_intensity(tape, window)
    half_spread = F.time_weighted(tape, 'spread', window)['median'] / 2
    pts = lam[(lam['delta'] >= (half_spread if np.isfinite(half_spread) else 0)) & (lam['both'] > 0)]
    if len(pts) < 2:
        return {'A': math.nan, 'k_intensity': math.nan, 'fit_points': len(pts)}
    duration = np.diff(F.bounds(tape, window))[0]
    counts = pts['both'].to_numpy() * duration
    slope, intercept = np.polyfit(pts['delta'], np.log(pts['both']), 1, w=np.sqrt(counts))
    return {'A': float(math.exp(intercept)), 'k_intensity': float(-slope), 'fit_points': len(pts)}


def fit_k_depth(tape: Tape, window: F.Window = None) -> dict[str, float]:
    """k from the geometric sweep depth past the touch, as `strategies/avellaneda_stoikov.py` estimates it online."""
    depth = F.taker_orders(tape, window)['depth']
    m = float(depth.mean()) if len(depth) else math.nan
    k = math.log(1 + 1 / m) if m and m > 0 else math.inf
    return {'mean_sweep_depth': m, 'k_depth': k}


def fit_all(tape: Tape, window: F.Window = None) -> dict[str, Any]:
    """Every fitted parameter for the window, JSON-serialisable."""
    ofi = F.ofi_regression(tape, (0.1, 1.0, 5.0), window).set_index('dt_s')
    imb = F.imbalance_response(tape, (1.0,), n_bins=10, window=window)
    mk = F.markouts(tape, np.array([0.1, 1.0, 10.0]), window).set_index('horizon_s')
    spread = F.time_weighted(tape, 'spread', window)
    out: dict[str, Any] = {
        'tape': tape.name,
        'window_s': list(F.bounds(tape, window)),
        'sigma_100ms': F.realized_vol(tape, 0.1, window),
        'sigma_1s': F.realized_vol(tape, 1.0, window),
        'spread_mean': spread['mean'],
        'spread_median': spread['median'],
        **fit_k_depth(tape, window),
        **fit_intensity(tape, window),
        'ofi_beta_1s': ofi.loc[1.0, 'beta'],
        'ofi_r2_1s': ofi.loc[1.0, 'r2'],
        'ofi_beta_next_1s': ofi.loc[1.0, 'beta_next'],
        'ofi_r2_next_1s': ofi.loc[1.0, 'r2_next'],
        'taker_pnl_1s': mk.loc[1.0, 'taker_pnl'] if 1.0 in mk.index else math.nan,
        'taker_pnl_10s': mk.loc[10.0, 'taker_pnl'] if 10.0 in mk.index else math.nan,
        'imbalance_dmid_1s': {f'{r.imb_mid:.2f}': round(float(r.mean_dmid), 4) for r in imb.itertuples()},
    }
    return {k: (None if isinstance(v, float) and not math.isfinite(v) else v) for k, v in out.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description='Fit strategy parameters from a market data CSV')
    parser.add_argument('csv', type=Path)
    parser.add_argument('--window', type=float, nargs=2, metavar=('T0_S', 'T1_S'), help='seconds since the first event')
    parser.add_argument('--refresh', action='store_true', help='ignore the replay cache')
    args = parser.parse_args()

    tape = load_tape(args.csv, refresh=args.refresh)
    params = fit_all(tape, tuple(args.window) if args.window else None)
    target = cache_path(args.csv).parent / f'{args.csv.stem}-params.json'
    target.write_text(json.dumps(params, indent=2))
    for key, value in params.items():
        if isinstance(value, float):
            print(f'{key:>22}: {value:.6g}')
        else:
            print(f'{key:>22}: {value}')
    print(f'\nwritten to {target}')


if __name__ == '__main__':
    main()
