"""Derived series and metrics for one `Run`.

PnL decomposition (full-logs runs). With mid m(t), inventory I(t) and fills
(side s, price p, qty q at time t_i), mark-to-mid PnL splits exactly into

    cash + I(t)·m(t) = Σ s·q·(m(t_i⁻) − p)   +   Σ s·q·(m(t) − m(t_i⁻))
                       └── spread edge ──┘       └─ inventory PnL ─┘

where m(t_i⁻) is the mid just before the fill. Edge is what each fill earned against the
mid at the moment it traded; inventory PnL is what holding the position earned or lost
afterwards (adverse selection shows up here). The simulator's PnL differs from the
mark-to-mid by the breach penalty and by liquidation cost (it marks against the
opposite side of the book, not the mid): liquidation = sim − mark-to-mid + penalty.

Markout at horizon h: s·(m(t_i + h) − p) per unit, volume-weighted, in ticks.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from results_explorer.load import Run

MARKOUT_HORIZONS_S = [0.0, 0.005, 0.05, 0.25, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0]


@dataclass
class Analysis:
    run: Run
    summary: dict[str, Any]  # flat metrics, JSON-serialisable
    grid: pd.DataFrame  # one row per dt: t_s, pnl (sim), drawdown, and, full logs: inventory, cash, mid, bid, ask, ...
    minute_pnl: pd.Series
    buckets: pd.DataFrame  # per bucket: pnl, and full logs: fills, volume, buy/sell volume, edge
    bucket_s: int
    fills: pd.DataFrame | None = None  # run.fills + t_s, mid_before, edge, inv_after, mo_<h>
    markouts: pd.DataFrame | None = None  # horizon_s, group, ticks (volume-weighted per unit), volume
    inventory_hist: pd.DataFrame | None = None  # inventory, seconds (time-weighted)
    orders: pd.DataFrame | None = None  # run.orders + lifetime_ms, placement_ticks
    extras: dict[str, Any] = field(default_factory=dict)


def _asof(t: np.ndarray, v: np.ndarray, q: np.ndarray, *, strict: bool = False, before: float = np.nan) -> np.ndarray:
    """v at the last t <= q (t < q if strict); `before` where there is none. `t` must be sorted."""
    idx = np.searchsorted(t, q, side='left' if strict else 'right') - 1
    out = v[np.clip(idx, 0, None)].astype('float64')
    out[idx < 0] = before
    return out


def _best_quotes(orders: pd.DataFrame, side: int, grid_us: np.ndarray) -> np.ndarray:
    """The user's best resting price on `side` at each grid time (NaN when none rests)."""
    o = orders[(orders['side'] == side) & orders['accept_us'].notna()].sort_values('accept_us')
    acc = o['accept_us'].to_numpy(dtype='int64')
    end = o['end_us'].to_numpy(dtype='int64')
    px = o['price'].to_numpy(dtype='float64')
    sign = -1.0 if side == 1 else 1.0  # heapq is a min-heap; bids want the max price
    heap: list[tuple[float, int]] = []
    out = np.full(len(grid_us), np.nan)
    j = 0
    for i, g in enumerate(grid_us):
        while j < len(acc) and acc[j] <= g:
            heapq.heappush(heap, (sign * px[j], int(end[j])))
            j += 1
        while heap and heap[0][1] <= g:
            heapq.heappop(heap)
        if heap:
            out[i] = sign * heap[0][0]
    return out


def _vw(values: np.ndarray, weights: np.ndarray) -> float:
    m = ~np.isnan(values)
    w = weights[m]
    return float(np.dot(values[m], w) / w.sum()) if w.sum() > 0 else float('nan')


def analyse(run: Run, dt: float = 1.0) -> Analysis:
    res = run.result
    duration = run.duration_s
    t_s = np.arange(0.0, np.floor(duration) + dt / 2, dt)
    grid_us = run.start_us + np.round(t_s * 1e6).astype('int64')

    pnl_t = run.pnl['elapsed_s'].to_numpy()
    pnl_v = run.pnl['pnl'].to_numpy(dtype='float64')
    grid = pd.DataFrame({'t_s': t_s, 'pnl': _asof(pnl_t, pnl_v, t_s, before=0.0)})
    grid['drawdown'] = grid['pnl'] - grid['pnl'].cummax()

    # Drawdown and per-minute statistics come from pnl.csv itself, not the grid.
    peak = np.maximum.accumulate(pnl_v)
    dd = pnl_v - peak
    trough = int(np.argmin(dd))
    peak_at = int(np.argmax(pnl_v[: trough + 1])) if trough > 0 else 0
    minute_t = np.arange(0, pnl_t[-1] + 1e-9, 60.0)
    minute_pnl = pd.Series(np.diff(_asof(pnl_t, pnl_v, minute_t, before=0.0)), index=minute_t[1:])
    mstd = float(minute_pnl.std(ddof=1)) if len(minute_pnl) > 1 else float('nan')
    mmean = float(minute_pnl.mean()) if len(minute_pnl) else float('nan')

    bucket_s = 300 if duration > 3600 else 60
    edges = np.arange(0, pnl_t[-1] + bucket_s, bucket_s)
    bpnl = np.diff(_asof(pnl_t, pnl_v, np.minimum(edges, pnl_t[-1]), before=0.0))
    buckets = pd.DataFrame({'start_s': edges[:-1], 'pnl': bpnl})

    hours = duration / 3600 if duration > 0 else float('nan')
    summary: dict[str, Any] = {
        'name': run.name,
        'strategy': run.metadata.get('trader_class', ''),
        'tape': str(run.metadata.get('data_file', '')).rsplit('/', 1)[-1],
        'full_logs': run.full_logs,
        'duration_s': round(duration, 3),
        'pnl': res.get('pnl'),
        'pnl_se': res.get('pnl_se'),
        'pnl_per_hour': res['pnl'] / hours if hours else None,
        'max_drawdown': float(dd.min()),
        'max_drawdown_from_s': float(pnl_t[peak_at]),
        'max_drawdown_to_s': float(pnl_t[trough]),
        'pnl_to_max_dd': res['pnl'] / -dd.min() if dd.min() < 0 else None,
        'minute_mean': mmean,
        'minute_std': mstd,
        'minute_sharpe': mmean / mstd if mstd else None,
        'minute_tstat': mmean / mstd * np.sqrt(len(minute_pnl)) if mstd else None,
        'minute_win_rate': float((minute_pnl > 0).mean()) if len(minute_pnl) else None,
        'best_minute': float(minute_pnl.max()) if len(minute_pnl) else None,
        'worst_minute': float(minute_pnl.min()) if len(minute_pnl) else None,
        'breach_penalty': res.get('breach_penalty'),
        'final_inventory': res.get('inventory'),
        'fills_reported': res.get('fills'),
        'traded_volume': res.get('traded_volume'),
        'abs_inventory_q50': res.get('abs_inventory_q50'),
        'abs_inventory_q95': res.get('abs_inventory_q95'),
        'abs_inventory_q997': res.get('abs_inventory_q997'),
    }
    analysis = Analysis(run=run, summary=summary, grid=grid, minute_pnl=minute_pnl, buckets=buckets, bucket_s=bucket_s)

    q = run.quotes
    if q is not None:
        qt = q['t_us'].to_numpy()
        grid['mid'] = _asof(qt, q['mid'].to_numpy(), grid_us)
        grid['bid'] = _asof(qt, q['bid'].to_numpy(), grid_us)
        grid['ask'] = _asof(qt, q['ask'].to_numpy(), grid_us)

    if not run.full_logs:
        return analysis
    _analyse_fills(analysis, grid_us)
    return analysis


def _analyse_fills(a: Analysis, grid_us: np.ndarray) -> None:
    run, grid, s = a.run, a.grid, a.summary
    limit = run.position_limit
    penalty_rate = float(run.metadata.get('position_breach_penalty_per_second', 10_000.0))
    f = run.fills.copy()
    f['t_s'] = (f['t_us'] - run.start_us) / 1e6
    ft = f['t_us'].to_numpy()
    sq = (f['side'] * f['qty']).to_numpy()
    f['inv_after'] = np.cumsum(sq)
    cash = np.cumsum(-sq * f['price'].to_numpy())

    # Inventory as a step function: value inv[k] holds from times[k] to times[k+1].
    times = np.concatenate([[run.start_us], ft])
    inv = np.concatenate([[0], f['inv_after'].to_numpy()])
    dur_s = np.diff(np.concatenate([times, [run.end_us]])) / 1e6
    breach = np.abs(inv) > limit
    cum_breach = np.concatenate([[0.0], np.cumsum(dur_s * breach)[:-1]])
    idx = np.searchsorted(times, grid_us, side='right') - 1
    grid['inventory'] = inv[idx]
    grid['cash'] = np.concatenate([[0.0], cash])[idx]
    grid['breach_s'] = cum_breach[idx] + breach[idx] * (grid_us - times[idx]) / 1e6

    hist = pd.DataFrame({'inventory': inv, 'seconds': dur_s}).groupby('inventory')['seconds'].sum().reset_index()
    a.inventory_hist = hist
    total_t = dur_s.sum()
    abs_inv = np.abs(inv)
    s.update(
        {
            'fills': len(f),
            'buy_volume': int(f.loc[f['side'] > 0, 'qty'].sum()),
            'sell_volume': int(f.loc[f['side'] < 0, 'qty'].sum()),
            'maker_volume_share': float(f.loc[f['maker'], 'qty'].sum() / f['qty'].sum()) if len(f) else None,
            'taker_fills': int((~f['maker']).sum()),
            'mean_fill_size': float(f['qty'].mean()) if len(f) else None,
            'pnl_per_unit_traded': s['pnl'] / f['qty'].sum() if len(f) else None,
            'mean_abs_inventory': float(np.dot(abs_inv, dur_s) / total_t),
            'mean_inventory': float(np.dot(inv, dur_s) / total_t),
            'time_at_limit_frac': float(dur_s[abs_inv >= limit].sum() / total_t),
            'time_flat_frac': float(dur_s[inv == 0].sum() / total_t),
            'breach_seconds': float((dur_s * breach).sum()),
            'max_long': int(inv.max()),
            'max_short': int(inv.min()),
            'inventory_zero_crossings': int(np.sum(np.diff(np.sign(inv[inv != 0])) != 0)),
        }
    )

    if run.quotes is not None:
        qt = run.quotes['t_us'].to_numpy()
        qmid = run.quotes['mid'].to_numpy()
        f['mid_before'] = _asof(qt, qmid, ft, strict=True)
        f['edge'] = f['side'] * (f['mid_before'] - f['price'])  # ticks per unit
        edge_cash = (f['edge'] * f['qty']).to_numpy()
        cum_edge = np.concatenate([[0.0], np.cumsum(edge_cash)])
        grid['edge'] = cum_edge[idx]
        grid['mtm'] = grid['cash'] + grid['inventory'] * grid['mid']
        grid['inventory_pnl'] = grid['mtm'] - grid['edge']
        grid['penalty'] = -penalty_rate * grid['breach_s']
        grid['liquidation'] = grid['pnl'] - grid['mtm'] - grid['penalty']

        rows = []
        w = f['qty'].to_numpy(dtype='float64')
        groups = {
            'all': np.ones(len(f), bool),
            'maker': f['maker'].to_numpy(),
            'taker': ~f['maker'].to_numpy(),
            'buys': (f['side'] > 0).to_numpy(),
            'sells': (f['side'] < 0).to_numpy(),
        }
        for h in MARKOUT_HORIZONS_S:
            target = ft + int(round(h * 1e6))
            m = _asof(qt, qmid, target, strict=(h == 0))
            m[target > run.end_us] = np.nan
            mo = f['side'].to_numpy() * (m - f['price'].to_numpy())
            f[f'mo_{h:g}'] = mo
            for g, mask in groups.items():
                if mask.any():
                    rows.append({'horizon_s': h, 'group': g, 'ticks': _vw(mo[mask], w[mask]), 'volume': float(w[mask].sum())})
        a.markouts = pd.DataFrame(rows)
        mk = a.markouts[a.markouts['group'] == 'all'].set_index('horizon_s')['ticks']
        last = grid.iloc[-1]
        s.update(
            {
                'edge_total': float(cum_edge[-1]),
                'edge_ticks_per_unit': float(mk.get(0.0, np.nan)),
                'markout_1s_ticks': float(mk.get(1.0, np.nan)),
                'markout_5s_ticks': float(mk.get(5.0, np.nan)),
                'markout_30s_ticks': float(mk.get(30.0, np.nan)),
                'markout_60s_ticks': float(mk.get(60.0, np.nan)),
                'inventory_pnl_total': float(last['inventory_pnl']),
                'liquidation_total': float(last['liquidation']),
                'mtm_pnl': float(last['mtm']),
            }
        )

        # Bucketed activity.
        b = (f['t_s'] // a.bucket_s * a.bucket_s).astype('int64')
        agg = f.assign(b=b, ev=edge_cash, bv=np.where(f['side'] > 0, f['qty'], 0), sv=np.where(f['side'] < 0, f['qty'], 0))
        agg = agg.groupby('b').agg(fills=('qty', 'size'), volume=('qty', 'sum'), buy_volume=('bv', 'sum'), sell_volume=('sv', 'sum'), edge=('ev', 'sum'))
        a.buckets = a.buckets.join(agg, on=a.buckets['start_s'].astype('int64')).fillna(0)

    # Orders.
    o = run.orders.copy()
    acc = o['accept_us'].notna()
    o['lifetime_ms'] = (o['end_us'] - o['accept_us']) / 1e3
    o['time_to_fill_ms'] = (o['first_fill_us'] - o['accept_us']) / 1e3
    if run.quotes is not None:
        mid_at_send = _asof(run.quotes['t_us'].to_numpy(), run.quotes['mid'].to_numpy(), o['sent_us'].to_numpy(dtype='int64'))
        o['placement_ticks'] = o['side'] * (mid_at_send - o['price'])  # > 0: behind the mid (passive side)
    a.orders = o
    oc = run.order_counts
    n_acc = int(acc.sum())
    filled_any = int((o['filled'] > 0).sum())
    s.update(
        {
            'orders_sent': oc.get('add_sent', 0),
            'orders_accepted': oc.get('add', 0),
            'orders_rejected': oc.get('add_failed', 0),
            'cancels_sent': oc.get('cancel_sent', 0),
            'cancels_failed': oc.get('cancel_failed', 0),
            'modifies_sent': oc.get('modify_sent', 0),
            'orders_with_fill_frac': filled_any / n_acc if n_acc else None,
            'qty_fill_ratio': float(o.loc[acc, 'filled'].sum() / o.loc[acc, 'qty'].sum()) if n_acc else None,
            'orders_per_fill': oc.get('add_sent', 0) / len(f) if len(f) else None,
            'median_order_lifetime_ms': float(o.loc[acc, 'lifetime_ms'].median()) if n_acc else None,
        }
    )

    # The user's best quotes against the market touch.
    ub = _best_quotes(o, 1, grid_us)
    ua = _best_quotes(o, -1, grid_us)
    grid['user_bid'], grid['user_ask'] = ub, ua
    s['bid_presence_frac'] = float(np.mean(~np.isnan(ub)))
    s['ask_presence_frac'] = float(np.mean(~np.isnan(ua)))
    s['two_sided_frac'] = float(np.mean(~np.isnan(ub) & ~np.isnan(ua)))
    if 'bid' in grid:
        grid['bid_depth'] = grid['bid'] - grid['user_bid']  # ticks behind the market's best bid (<0: inside)
        grid['ask_depth'] = grid['user_ask'] - grid['ask']
        s['bid_at_or_inside_touch_frac'] = float((grid['bid_depth'] <= 0).sum() / max(1, grid['bid_depth'].notna().sum()))
        s['ask_at_or_inside_touch_frac'] = float((grid['ask_depth'] <= 0).sum() / max(1, grid['ask_depth'].notna().sum()))
    a.fills = f
