"""Historical execution diagnostics, not a counterfactual strategy-profit model.

Queue cohorts use actual, non-crossing G orders and their actual FIFO executions.
Cancellations are competing outcomes, not missing observations. End-of-window
orders without a complete follow-up horizon are excluded, never counted as failures.
Maker markouts use actual maker prices and information before each taker event.
Cancel experiments request a cancel after a specified order age and count fills
strictly before its hypothetical activation, matching the simulator's boundary rule.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.features import Window, bounds
from analysis.replay import Tape

QUEUE_LABELS = ['0', '1-5', '6-20', '21-100', '>100']


def _window_us(tape: Tape, window: Window):
    a, b = bounds(tape, window)
    return tape.start_us + round(a * 1e6), tape.start_us + round(b * 1e6)


def _size_mask(volume, size):
    ranges = {'all': (1, np.inf), '1-5': (1, 5), '6-20': (6, 20),
              '21-100': (21, 100), '>100': (101, np.inf)}
    lo, hi = ranges[size]
    return (volume >= lo) & (volume <= hi)


def passive_orders(tape, window=None, side='both', size='all', placement='touch'):
    a, b = _window_us(tape, window)
    o = tape.orders
    if 'queue_ahead' not in o:
        raise ValueError('Replay cache is out of date; restart the explorer with --refresh.')
    keep = ~o.ioc & ~o.crossed & (o.queue_ahead >= 0) & o.t_us.between(a, b)
    keep &= _size_mask(o.volume, size)
    if side != 'both':
        keep &= o.side == side
    if placement == 'touch':
        keep &= o.dist == 0
    elif placement == 'inside':
        keep &= o.dist > 0
    elif placement == 'behind':
        keep &= o.dist < 0
    return o.loc[keep].copy()


def queue_fills(tape, window=None, horizon_s=1.0, side='both', size='all', placement='touch'):
    if horizon_s <= 0:
        raise ValueError('Fill horizon must be positive')
    _, end = _window_us(tape, window)
    o = passive_orders(tape, window, side, size, placement)
    deadline = o.t_us + round(horizon_s * 1e6)
    excluded = int((deadline > end).sum())
    o = o.loc[deadline <= end].set_index('row')
    tr = tape.trades
    tr = tr.loc[tr.maker_row.isin(o.index)]
    birth = tape.events.t_us.to_numpy()[tr.maker_row.to_numpy(dtype=int)]
    tr = tr.loc[tr.t_us.to_numpy() <= birth + round(horizon_s * 1e6)]
    filled = tr.groupby('maker_row').qty.sum()
    first = tr.groupby('maker_row').t_us.min()
    o['filled_by_horizon'] = filled.reindex(o.index).fillna(0).astype('int64')
    o['first_fill_ms'] = (first.reindex(o.index) - o.t_us) / 1000
    o['any_fill'] = o.filled_by_horizon > 0
    o['full_fill'] = o.filled_by_horizon >= o.volume
    o['fraction'] = o.filled_by_horizon / o.volume
    o['queue_bucket'] = pd.cut(o.queue_ahead, [-1, 0, 5, 20, 100, np.inf], labels=QUEUE_LABELS)
    rows = []
    for label, group in o.groupby('queue_bucket', observed=True, sort=True):
        rows.append(dict(queue=str(label), orders=len(group), any_fill=group.any_fill.mean(),
                         full_fill=group.full_fill.mean(), mean_fraction=group.fraction.mean(),
                         median_first_ms=group.first_fill_ms.median()))
    stats = pd.DataFrame(rows, columns=['queue', 'orders', 'any_fill', 'full_fill', 'mean_fraction', 'median_first_ms'])
    return stats, dict(orders=len(o), excluded_tail=excluded,
                       any_fill=float(o.any_fill.mean()) if len(o) else np.nan,
                       full_fill=float(o.full_fill.mean()) if len(o) else np.nan)


def _quote_values(tape, times, column='mid'):
    """Last observed state, including one-sided gaps (not last two-sided state)."""
    q = tape.quotes
    indices = np.searchsorted(q.t_us.to_numpy(), times, side='right') - 1
    out = np.full(len(indices), np.nan)
    good = (indices >= 0) & (times <= tape.end_us)
    values = q[column].to_numpy(dtype=float)
    out[good] = values[indices[good]]
    # All execution diagnostics require a genuinely two-sided, uncrossed book.
    valid = q.mid.notna().to_numpy() & (q.spread.to_numpy() > 0)
    out[good] = np.where(valid[indices[good]], out[good], np.nan)
    return out


def _quantile(values, weights, p):
    order = np.argsort(values, kind='stable')
    cumulative = np.cumsum(weights[order])
    return float(values[order[np.searchsorted(cumulative, p * cumulative[-1], side='left')]])


def _markout_summary(group):
    y = group.markout.to_numpy()
    w = group.qty.to_numpy(dtype=float)
    return dict(fills=len(group), takers=group.taker_row.nunique(), lots=int(w.sum()),
                mean=float(np.average(y, weights=w)), p10=_quantile(y, w, .1),
                median=_quantile(y, w, .5), p90=_quantile(y, w, .9),
                adverse=float(w[y < 0].sum() / w.sum()))


def maker_markouts(tape, window=None, horizon_s=1.0, side='both', size='all'):
    if horizon_s <= 0:
        raise ValueError('Markout horizon must be positive')
    a, b = _window_us(tape, window)
    tr = tape.trades
    selected = tr.loc[tr.t_us.between(a, b)].copy()
    selected['side'] = np.where(selected.aggressor == 'A', 'B', 'A')
    if side != 'both':
        selected = selected.loc[selected.side == side]
    selected = selected.loc[_size_mask(
        tape.events.volume.to_numpy()[selected.maker_row.to_numpy(dtype=int)], size)]
    future = selected.t_us.to_numpy() + round(horizon_s * 1e6)
    excluded_tail = int((future > b).sum())
    selected = selected.loc[future <= b].copy()
    columns = ['group', 'fills', 'takers', 'lots', 'mean', 'p10', 'median', 'p90', 'adverse']
    if selected.empty:
        return {key: pd.DataFrame(columns=columns) for key in ('spread', 'imbalance', 'volatility', 'aggressor_size')}, dict(fills=0, excluded_tail=excluded_tail)
    pre_row = selected.taker_row.to_numpy(dtype=int) - 1
    # State before the specific taker EVENT, not after another event at the same timestamp.
    pre = tape.events.iloc[np.maximum(pre_row, 0)]
    bid, ask = pre.bid.to_numpy(), pre.ask.to_numpy()
    bq, aq = pre.bid_qty.to_numpy(dtype=float), pre.ask_qty.to_numpy(dtype=float)
    midpoint = _quote_values(tape, selected.t_us.to_numpy() + round(horizon_s * 1e6))
    good = (pre_row >= 0) & (bid > 0) & (ask > bid) & np.isfinite(midpoint)
    sign = np.where(selected.side == 'B', 1, -1)
    selected['markout'] = sign * (midpoint - selected.price.to_numpy())
    selected['spread'] = pd.cut(ask - bid, [0, 1, 2, 5, np.inf], labels=['1', '2', '3-5', '>5'])
    imbalance = np.divide(bq, bq + aq, out=np.full(len(bq), np.nan), where=(bq + aq) > 0)
    relative = np.where(sign == 1, imbalance, 1 - imbalance)
    selected['imbalance'] = pd.cut(relative, [-np.inf, .2, .4, .6, .8, np.inf],
                                    labels=['0-.2', '.2-.4', '.4-.6', '.6-.8', '.8-1'])
    submitted = tape.events.volume.to_numpy()[selected.taker_row.to_numpy(dtype=int)]
    selected['aggressor_size'] = pd.cut(submitted, [0, 5, 20, 100, np.inf], labels=['1-5', '6-20', '21-100', '>100'])
    # Causal 30-second RMS of 1-second mid changes, available strictly before the fill.
    grid = np.arange(tape.start_us, tape.end_us + 1, 1_000_000, dtype=np.int64)
    mids = _quote_values(tape, grid)
    sigma = pd.Series(mids).diff().pow(2).rolling(30, min_periods=5).mean().pow(.5).to_numpy()
    idx = np.searchsorted(grid, selected.t_us.to_numpy(), side='left') - 1
    vol = np.full(len(idx), np.nan)
    valid = idx >= 0
    vol[valid] = sigma[idx[valid]]
    finite = vol[np.isfinite(vol) & good]
    low, high = np.quantile(finite, [.333333, .666667]) if len(finite) else (np.nan, np.nan)
    regimes = np.where(vol <= low, 'Low', np.where(vol <= high, 'Medium', 'High')).astype(object)
    regimes[~np.isfinite(vol)] = 'No history'
    selected['volatility'] = pd.Categorical(regimes, categories=['Low', 'Medium', 'High', 'No history'], ordered=True)
    invalid = int((~good).sum())
    selected = selected.loc[good]
    tables = {}
    for field in ('spread', 'imbalance', 'volatility', 'aggressor_size'):
        tables[field] = pd.DataFrame([
            dict(group=str(label), **_markout_summary(g))
            for label, g in selected.groupby(field, observed=True, sort=True)
        ], columns=columns)
    summary = _markout_summary(selected) if len(selected) else dict(fills=0)
    summary.update(excluded_tail=excluded_tail, invalid_quotes=invalid, vol_low=low, vol_high=high)
    return tables, summary


def latency_risk(tape, window=None, latency_ms=5.0, cancel_age_ms=100.0, side='both',
                 size='all', placement='touch', sample_s=.1):
    if latency_ms <= 0 or cancel_age_ms < 0 or sample_s <= 0:
        raise ValueError('Latency and sampling must be positive; cancel age cannot be negative')
    delays = sorted(set([1.0, 5.0, 10.0, 25.0, float(latency_ms)]))
    a, b = _window_us(tape, window)
    # Same complete-follow-up cohort for all delays. Bound chart cost on long tapes.
    spacing = max(sample_s, max(0, (b-a)/1e6) / 50_000)
    latest = b - round(max(delays) * 1000)
    t = np.arange(a, latest + 1, max(1, round(spacing * 1e6)), dtype=np.int64)
    mid = _quote_values(tape, t)
    selected_sides = ['B', 'A'] if side == 'both' else [side]
    quote_rows = []
    for direction in selected_sides:
        price = _quote_values(tape, t, 'bid' if direction == 'B' else 'ask')
        sign = 1 if direction == 'B' else -1
        for delay in delays:
            future_t = t + round(delay * 1000)
            after = _quote_values(tape, future_t)
            opposite = _quote_values(tape, future_t, 'ask' if direction == 'B' else 'bid')
            good = np.isfinite(mid) & np.isfinite(after) & np.isfinite(price) & np.isfinite(opposite)
            moves = sign * (after[good] - mid[good])
            crossing = price[good] >= opposite[good] if direction == 'B' else price[good] <= opposite[good]
            quote_rows.append(dict(side=direction, delay_ms=delay, samples=int(good.sum()),
                                   adverse=float(np.mean(moves < 0)) if len(moves) else np.nan,
                                   marketable=float(np.mean(crossing)) if len(moves) else np.nan,
                                   mean_adverse_ticks=float(np.maximum(-moves, 0).mean()) if len(moves) else np.nan))

    o = passive_orders(tape, window, side, size, placement)
    age_us = round(cancel_age_ms * 1000)
    request = o.t_us + age_us
    # Orders actually alive after the request timestamp; full common follow-up required.
    live = o.end_t_us.isna() | (o.end_t_us > request)
    o = o.loc[live & (request + round(max(delays)*1000) <= b)].set_index('row')
    tr = tape.trades.loc[tape.trades.maker_row.isin(o.index)]
    age = tr.t_us.to_numpy() - tape.events.t_us.to_numpy()[tr.maker_row.to_numpy(dtype=int)]
    past = tr.loc[age <= age_us].groupby('maker_row').qty.sum()
    remaining = o.volume - past.reindex(o.index).fillna(0)
    cancel_rows = []
    for direction in selected_sides:
        cohort = o.loc[o.side == direction]
        rem = remaining.reindex(cohort.index)
        for delay in delays:
            pending = tr.loc[(age > age_us) & (age < age_us + round(delay * 1000))]
            qty = pending.groupby('maker_row').qty.sum().reindex(cohort.index).fillna(0)
            n = len(cohort)
            cancel_rows.append(dict(side=direction, delay_ms=delay, orders=n,
                                    any_fill=float((qty > 0).mean()) if n else np.nan,
                                    full_fill=float((qty >= rem).mean()) if n else np.nan,
                                    mean_lost_lots=float(qty.mean()) if n else np.nan,
                                    lost_lots=int(qty.sum())))
    return pd.DataFrame(quote_rows), pd.DataFrame(cancel_rows), dict(sample_s=spacing, cancel_orders=len(o))
