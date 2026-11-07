"""Fit the round-1 generator model to a tape and write its parameters (and book statistics) to JSON.

Usage:
    uv run python strategies/peter/r1_model.py data/round1.csv
    uv run python strategies/peter/r1_model.py data/synth/r1_seed1.csv --out data/synth/r1_seed1_fit.json

The model (see round1_generator.md for the evidence):

    S(t)     hidden fair price, a pure jump process: S0 + c * sum over takers of sign * sqrt(filled)
    makers   Poisson(maker_rate), side 50/50, size round(lognormal), bid = min(floor(S - D), ask - 1),
             ask = max(ceil(S + D), bid + 1), D = d_shift + Exp(d_scale); a cancel decision arrives with
             hazard alpha + beta * max(0, adverse move of S since the insert), the cancel row Exp(life_b)
             later, whether or not the order has been filled
    takers   Poisson(taker_rate) per side, size round(lognormal), IOC at round(S + 1) (buy) /
             round(S - 1) (sell), S taken before the taker's own jump

Every step re-derives the numbers from the tape, so the script also checks whether a new round
still follows the model (the S identification reports how many orders violate the price rules).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.cache import load_tape  # noqa: E402


# ------------------------------------------------------------------ data
def order_table(tape) -> pd.DataFrame:
    """One row per insert with the book just before it and its cancel time (if any)."""
    ev, o = tape.events, tape.orders.copy()
    prev = np.maximum(o.row.to_numpy() - 1, 0)
    bid = ev.bid.to_numpy()[prev].astype(float)
    ask = ev.ask.to_numpy()[prev].astype(float)
    bid[(bid == 0) | (o.row.to_numpy() == 0)] = np.nan
    ask[(ask == 0) | (o.row.to_numpy() == 0)] = np.nan
    o['bid'], o['ask'] = bid, ask
    c = ev.loc[ev.op == 'C', ['t_us', 'order_id']].rename(columns={'t_us': 'cancel_us'})
    o = o.merge(c, on='order_id', how='left')
    o['is_buy'] = (o.side == 'B').to_numpy()
    return o


# ------------------------------------------------------------------ arrivals
def fit_arrivals(o: pd.DataFrame, duration: float) -> dict:
    g, f = o[~o.ioc], o[o.ioc]
    out = {'maker_rate': len(g) / duration, 'taker_rate_per_side': len(f) / duration / 2,
           'maker_buy_share': float(g.is_buy.mean()), 'taker_buy_share': float(f.is_buy.mean())}
    checks = {}
    for name, ts in [('maker', g.t_us), ('taker_buy', f[f.is_buy].t_us), ('taker_sell', f[~f.is_buy].t_us)]:
        gaps = np.diff(ts.to_numpy()) / 1e6
        cnt = np.bincount((ts.to_numpy() // 10_000_000).astype(int))[:-1]
        checks[name] = {'gap_cv': float(gaps.std() / gaps.mean()), 'dispersion_10s': float(cnt.var() / cnt.mean())}
    sgn = np.where(f.is_buy, 1, -1)
    checks['taker_sign_acf1'] = float(np.corrcoef(sgn[:-1], sgn[1:])[0, 1])
    out['checks'] = checks
    return out


# ------------------------------------------------------------------ sizes
def fit_round_lognormal(v: np.ndarray) -> dict:
    """MLE of v = max(1, round(exp(N(mu, sd))))."""
    cnt = np.bincount(v.astype(int))
    k = np.arange(len(cnt), dtype=float)
    obs = cnt > 0
    lo = np.log(np.maximum(k - 0.5, 1e-300)); lo[k <= 1] = -np.inf
    hi = np.log(k + 0.5)

    def nll(th):
        mu, sd = th
        if sd <= 0:
            return 1e18
        p = stats.norm.cdf((hi - mu) / sd) - stats.norm.cdf((lo - mu) / sd)
        return -(cnt[obs] * np.log(np.maximum(p[obs], 1e-300))).sum()

    lv = np.log(v)
    r = optimize.minimize(nll, [lv.mean(), lv.std()], method='Nelder-Mead', options={'xatol': 1e-6, 'fatol': 1e-6})
    return {'mu': float(r.x[0]), 'sd': float(r.x[1])}


# ------------------------------------------------------------------ lifetimes
def fit_lifetimes(life: np.ndarray) -> dict:
    """MLE of life = Exp(rate a) + Exp(rate b) (hypoexponential), a < b."""
    life = life[life > 0]

    def nll(th):
        a, b = np.exp(th)
        if abs(a - b) < 1e-9:
            b = a + 1e-6
        dens = a * b / (b - a) * (np.exp(-a * life) - np.exp(-b * life))
        return -np.log(np.maximum(dens, 1e-300)).sum()

    r = optimize.minimize(nll, np.log([1 / life.mean() * 1.1, 20.0]), method='Nelder-Mead')
    a, b = sorted(np.exp(r.x))
    ks_hypo = stats.kstest(life[::7], lambda x: 1 - (b * np.exp(-a * x) - a * np.exp(-b * x)) / (b - a)).statistic
    g = stats.gamma.fit(life[::7], floc=0)
    ks_gamma = stats.kstest(life[::7], 'gamma', args=g).statistic
    return {'life_a': float(a), 'life_b': float(b), 'mean_s': float(life.mean()),
            'ks_hypoexp': float(ks_hypo), 'ks_gamma': float(ks_gamma)}


# ------------------------------------------------------------------ hidden price S
def impact_units(o: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """For each insert: sum of sign*sqrt(filled) over takers strictly before it (S before the order)."""
    f = o[o.ioc]
    jt = f.t_us.to_numpy()
    cu = np.cumsum(np.where(f.is_buy, 1.0, -1.0) * np.sqrt(f.filled.to_numpy()))
    k = np.searchsorted(jt, o.t_us.to_numpy(), 'left') - 1
    return np.where(k >= 0, cu[np.maximum(k, 0)], 0.0), cu


def s0_bounds(c: float, o: pd.DataFrame, U: np.ndarray, taker_off: float = 1.0) -> tuple[float, float]:
    """Feasible interval for S0 given c: maker bids <= S <= maker asks, taker price - S in off +- 0.5."""
    p = o.price.to_numpy().astype(float) - c * U
    g, b = ~o.ioc.to_numpy(), o.is_buy.to_numpy()
    lo = [p[g & b].max(), (p[~g & b] - taker_off - 0.5).max(), (p[~g & ~b] + taker_off - 0.5).max()]
    hi = [p[g & ~b].min(), (p[~g & b] - taker_off + 0.5).min(), (p[~g & ~b] + taker_off + 0.5).min()]
    return max(lo), min(hi)


def identify_s(o: pd.DataFrame) -> dict:
    U, _ = impact_units(o)
    cs = np.arange(0.005, 0.06, 0.0005)
    width = np.array([np.subtract(*s0_bounds(c, o, U)[::-1]) for c in cs])
    c0 = cs[np.argmax(width)]
    fine = np.arange(c0 - 0.001, c0 + 0.001, 0.000005)
    wf = np.array([np.subtract(*s0_bounds(c, o, U)[::-1]) for c in fine])
    feas = fine[wf >= 0]
    c = float(fine[np.argmax(wf)])
    lo, hi = s0_bounds(c, o, U)
    s0 = (lo + hi) / 2
    S = s0 + c * U
    x = np.where(o.is_buy, 1, -1) * (o.price.to_numpy() - S)
    g = ~o.ioc.to_numpy()
    return {
        'c': c, 's0': float(s0), 's0_interval': [float(lo), float(hi)],
        'c_feasible': [float(feas.min()), float(feas.max())] if len(feas) else None,
        'maker_violations': int((x[g] > 1e-9).sum()),
        'taker_violations': int(((x[~g] < 0.5 - 1e-9) | (x[~g] > 1.5 + 1e-9)).sum()),
        'n_orders': int(len(o)),
    }


def fit_offset(o: pd.DataFrame, S: np.ndarray) -> dict:
    """Exact MLE of D = shift + Exp(scale) from maker prices, with the no-cross clip."""
    g = ~o.ioc.to_numpy()
    b = o.is_buy.to_numpy()[g]
    p = o.price.to_numpy()[g].astype(float)
    s = S[g]
    lim = np.where(b, o.ask.to_numpy()[g] - 1, o.bid.to_numpy()[g] + 1)
    y = np.where(b, s - p, p - s)                       # = D + fractional part, >= D
    ylim = np.where(b, s - lim, lim - s)                # distance from S to the clip price
    clipped = np.isclose(p, lim)                        # price at the clip: floor(S-D) >= lim, i.e. D <= ylim
    ok = ~np.isnan(lim)
    y, ylim, clipped = y[ok], ylim[ok], clipped[ok]

    def cdf(z, a, sc):
        return np.where(z > a, 1 - np.exp(-(z - a) / sc), 0.0)

    def nll(th):
        a, sc = th
        if sc <= 0 or a < 0 or a > y.min() + 1e-9:
            return 1e18
        pr = np.where(clipped, cdf(ylim, a, sc), cdf(y, a, sc) - cdf(y - 1, a, sc))
        return -np.log(np.maximum(pr, 1e-300)).sum()

    r = optimize.minimize(nll, [min(0.04, y.min() * 0.9), 0.45], method='Nelder-Mead', options={'xatol': 1e-6, 'fatol': 1e-4})
    r0 = optimize.minimize_scalar(lambda sc: nll([0.0, sc]), bounds=(0.05, 3), method='bounded')
    return {'d_shift': float(r.x[0]), 'd_scale': float(r.x[1]), 'min_gap_S_minus_price': float(y.min()),
            'at_clip_price_share': float(clipped.mean()),  # price one tick inside the opposite touch (censored in the fit)
'loglik_gain_vs_plain_exp_per_order': float((r0.fun - r.fun) / len(y))}


# ------------------------------------------------------------------ state-dependent cancels
# Generator values found by matching round-1 statistics with r1_synth.py (the measured hazards below are lower:
# they see the decision through the Exp(life_b) delay and lose orders to fills).
CANCEL_CALIBRATED = {'cancel_alpha': 1.72, 'cancel_beta': 5.8, 'life_b': 40.0}


def cancel_hazard(o: pd.DataFrame, S_fn, n_orders: int = 150_000, dt_us: int = 5_000) -> dict:
    """Cancel hazard (1/s) of live maker orders by mv = adverse move of S since the insert, and by placement.

    mv > 0 means S moved toward / through the order (down for a bid, up for an ask).
    """
    g = o[(~o.ioc) & o.cancel_us.notna()]
    g = g.sample(min(len(g), n_orders), random_state=0)
    end = np.where(g.end == 'filled', g.end_t_us, g.cancel_us).astype(np.int64)  # exposure ends at fill or cancel
    t0 = g.t_us.to_numpy()
    n = ((end - t0) // dt_us + 1).astype(int)
    rep = np.repeat(np.arange(len(g)), n)
    step = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    tt = t0[rep] + step * dt_us
    sg = np.where(g.is_buy.to_numpy(), 1.0, -1.0)[rep]
    s_ins = S_fn(t0)[rep]
    mv = -sg * (S_fn(tt) - s_ins)
    d0 = sg * (g.price.to_numpy()[rep] - s_ins)
    canc = (step == n[rep] - 1) & (g.end.to_numpy()[rep] != 'filled')
    old = step * dt_us > 50_000
    per_s = 1e6 / dt_us
    edges = [-9, 0.0001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 9]
    hz_mv = pd.Series(canc[old]).groupby(pd.cut(mv[old], edges)).mean() * per_s
    flat = old & (mv <= 0.0001)
    hz_d0 = pd.Series(canc[flat]).groupby(pd.cut(d0[flat], [-9, -2, -1.5, -1, -0.5, 0.0001])).mean() * per_s
    mid = (mv[old] > 0.0001) & (mv[old] < 1.5)
    slope = np.polyfit(mv[old][mid], canc[old][mid] * per_s, 1)
    return {
        'hazard_by_mv': {str(k): round(float(v), 3) for k, v in hz_mv.items()},
        'hazard_by_insert_distance_when_mv<=0': {str(k): round(float(v), 3) for k, v in hz_d0.items()},
        'measured_linear_fit_mv>0': {'intercept': float(slope[1]), 'slope': float(slope[0])},
        'hazard_age<50ms_mv<=0': float(canc[(~old) & (mv <= 0.0001)].mean() * per_s),
        **CANCEL_CALIBRATED,
    }


# ------------------------------------------------------------------ book statistics (for validating synthetic tapes)
def book_stats(tape, o: pd.DataFrame, S_fn) -> dict:
    ev = tape.events
    tu = ev.t_us.to_numpy()
    dur = np.diff(np.r_[tu, tu[-1]]) / 1e6
    bid = ev.bid.to_numpy().astype(float); ask = ev.ask.to_numpy().astype(float)
    two = (bid > 0) & (ask > 0)
    spread = np.where(two, ask - bid, np.nan)
    T = dur[two].sum()
    sp = {str(int(k)): float(dur[two & (spread == k)].sum() / T) for k in (1, 2, 3)}
    w = dur[two] / T
    bq, aq = ev.bid_qty.to_numpy()[two], ev.ask_qty.to_numpy()[two]
    mid = np.where(two, (bid + ask) / 2, np.nan)
    Se = S_fn(tu)
    grid = np.arange(tu[0], tu[-1], 1_000_000)
    k = np.searchsorted(tu, grid, 'right') - 1
    mg = pd.Series(mid[k]).ffill().bfill().to_numpy()
    d1 = np.diff(mg)
    tr = tape.trades
    f = o[o.ioc]
    return {
        'spread_time_share': sp,
        'touch_qty_mean': float((w * (bq + aq) / 2).sum()),
        'touch_qty_median': float(np.median(np.r_[bq, aq])),
        'vol_1s_ticks': float(d1.std()),
        'mid_minus_S_sd': float(np.sqrt(np.nansum(dur * (mid - Se) ** 2) / np.nansum(dur * ~np.isnan(mid)))),
        'trades_per_s': float(len(tr) / tape.duration_s),
        'taker_fill_share_of_volume': float(f.filled.sum() / f.volume.sum()),
        'takers_with_no_fill_share': float((f.filled == 0).mean()),
        'cancels_of_filled_orders_share': float(((o.end == 'filled') & o.cancel_us.notna()).sum() / o.cancel_us.notna().sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('csv', type=Path)
    ap.add_argument('--out', type=Path, default=None, help='default: strategies/peter/tuned/<stem>_model.json')
    args = ap.parse_args()
    out = args.out or Path(__file__).parent / 'tuned' / f'{args.csv.stem}_model.json'

    tape = load_tape(args.csv)
    o = order_table(tape)
    duration = tape.duration_s
    res: dict = {'source': str(args.csv), 'duration_s': duration}
    print('arrivals ...', flush=True)
    res['arrivals'] = fit_arrivals(o, duration)
    print('sizes ...', flush=True)
    res['maker_size'] = fit_round_lognormal(o[~o.ioc].volume.to_numpy())
    res['taker_size'] = fit_round_lognormal(o[o.ioc].volume.to_numpy())
    print('lifetimes ...', flush=True)
    g = o[~o.ioc]
    res['lifetime'] = fit_lifetimes(((g.cancel_us - g.t_us) / 1e6).dropna().to_numpy())
    res['lifetime']['share_of_makers_cancelled'] = float(g.cancel_us.notna().mean())
    print('hidden price ...', flush=True)
    res['S'] = identify_s(o)
    c, s0 = res['S']['c'], res['S']['s0']
    U, cu = impact_units(o)
    S = s0 + c * U
    res['maker_offset'] = fit_offset(o, S)
    f = o[o.ioc]
    jt = f.t_us.to_numpy()

    def S_after(tus):
        k = np.searchsorted(jt, tus, 'right') - 1
        return s0 + c * np.where(k >= 0, cu[np.maximum(k, 0)], 0.0)

    print('cancel hazard ...', flush=True)
    res['cancel'] = cancel_hazard(o, S_after)
    print('book stats ...', flush=True)
    res['book'] = book_stats(tape, o, S_after)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    print(f'-> {out}')


if __name__ == '__main__':
    main()
