"""Statistics of a replayed tape: prices on a time grid, volatility, liquidity, flow and predictability.

Every function takes the `Tape` and an optional `window = (t0_s, t1_s)` on the
round clock (seconds since the first event), and is pure numpy/pandas, so the
dashboard, notebooks and `analysis.fit` share one definition of each number.

Units: prices and price moves in ticks, time in seconds, volatility in ticks
per sqrt-second.

The book state between events is the state after the last event (as-of). Mid
and imbalance are only defined with both sides present. The tapes often cancel
and re-insert a quote a microsecond apart, which leaves a side empty for that
microsecond, so the grid samplers carry the last two-sided value over such gaps.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.replay import Tape

Window = tuple[float, float] | None


# --------------------------------------------------------------------------- helpers


def bounds(tape: Tape, window: Window) -> tuple[float, float]:
    """The window in round seconds, defaulting to the whole tape."""
    if window is None:
        return 0.0, tape.duration_s
    return max(0.0, float(window[0])), min(tape.duration_s, float(window[1]))


def _in(t_s: pd.Series | np.ndarray, window: tuple[float, float]) -> np.ndarray:
    t = np.asarray(t_s)
    return (t >= window[0]) & (t <= window[1])


def trades_s(tape: Tape, window: Window = None) -> pd.DataFrame:
    """Trades with `t_s` and `sign` (+1 buyer-initiated, -1 seller-initiated), restricted to the window."""
    tr = tape.trades.copy()
    tr['t_s'] = (tr['t_us'] - tape.start_us) / 1e6
    tr['sign'] = np.where(tr['aggressor'] == 'B', 1, -1)
    return tr[_in(tr['t_s'], bounds(tape, window))].reset_index(drop=True)


def taker_orders(tape: Tape, window: Window = None) -> pd.DataFrame:
    """One row per aggressive order that traded: first fill time, sign, total qty, levels swept, worst price."""
    tr = trades_s(tape, window)
    if tr.empty:
        return pd.DataFrame(columns=['t_s', 't_us', 'sign', 'qty', 'depth', 'worst_price', 'n_fills'])
    g = tr.groupby('taker_row', sort=True)
    out = pd.DataFrame(
        {
            't_s': g['t_s'].first(),
            't_us': g['t_us'].first(),
            'sign': g['sign'].first(),
            'qty': g['qty'].sum(),
            'depth': g['depth'].max(),
            'worst_price': np.where(g['sign'].first() > 0, g['price'].max(), g['price'].min()),
            'n_fills': g.size(),
        }
    )
    return out.sort_values('t_us').reset_index()


def _two_sided(tape: Tape) -> pd.DataFrame:
    q = tape.quotes
    return q[q['mid'].notna()]


def asof(tape: Tape, t_s: np.ndarray, column: str = 'mid', *, strict: bool = False) -> np.ndarray:
    """`column` of the last two-sided quote at (or, with `strict`, before) each time; NaN before the first."""
    q = _two_sided(tape)
    idx = np.searchsorted(q['t_s'].to_numpy(), t_s, side='left' if strict else 'right') - 1
    values = q[column].to_numpy(dtype='float64')
    out = np.full(len(idx), np.nan)
    ok = idx >= 0
    out[ok] = values[idx[ok]]
    return out


def grid(tape: Tape, dt_s: float, window: Window = None) -> np.ndarray:
    """Times t0, t0 + dt, ... up to t1."""
    t0, t1 = bounds(tape, window)
    return np.arange(t0, t1 + 1e-9, dt_s)


def mid_on_grid(tape: Tape, dt_s: float, window: Window = None) -> pd.Series:
    """Mid price sampled every `dt_s` seconds (as-of), indexed by round seconds."""
    t = grid(tape, dt_s, window)
    return pd.Series(asof(tape, t), index=pd.Index(t, name='t_s'), name='mid')


# --------------------------------------------------------------------------- overview


def summary(tape: Tape, window: Window = None) -> dict[str, float]:
    """The headline numbers for the overview card and the round comparison table."""
    w = bounds(tape, window)
    duration = max(w[1] - w[0], 1e-9)
    ev_t = (tape.events['t_us'].to_numpy() - tape.start_us) / 1e6
    ev = tape.events[_in(ev_t, w)]
    orders = tape.orders[_in((tape.orders['t_us'] - tape.start_us) / 1e6, w)]
    tr = trades_s(tape, w)
    takers = taker_orders(tape, w)
    spread = time_weighted(tape, 'spread', w)
    mid = mid_on_grid(tape, 1.0, w).dropna()
    one_sided = one_sided_share(tape, w)

    return {
        'duration_s': duration,
        'events': len(ev),
        'events_per_s': len(ev) / duration,
        'inserts': int((ev['op'] == 'I').sum()),
        'cancels': int((ev['op'] == 'C').sum()),
        'ioc_inserts': int(orders['ioc'].sum()),
        'cancel_to_insert': (ev['op'] == 'C').sum() / max((ev['op'] == 'I').sum(), 1),
        'trades': len(tr),
        'taker_orders': len(takers),
        'trades_per_min': len(takers) / duration * 60,
        'traded_volume': int(tr['qty'].sum()),
        'buy_volume_share': tr.loc[tr['sign'] > 0, 'qty'].sum() / max(tr['qty'].sum(), 1),
        'passive_fill_rate': float((orders.loc[~orders['crossed'], 'filled'] > 0).mean()) if (~orders['crossed']).any() else np.nan,
        'price_min': float(np.nanmin(mid)) if len(mid) else np.nan,
        'price_max': float(np.nanmax(mid)) if len(mid) else np.nan,
        'mid_start': float(mid.iloc[0]) if len(mid) else np.nan,
        'mid_end': float(mid.iloc[-1]) if len(mid) else np.nan,
        'spread_mean': spread['mean'],
        'spread_median': spread['median'],
        'spread_1tick_share': spread['share_at_min'] if spread['min'] == 1 else 0.0,
        'spread_min': spread['min'],
        'one_sided_share': one_sided,
        'vol_100ms': realized_vol(tape, 0.1, w),
        'vol_1s': realized_vol(tape, 1.0, w),
        'vol_10s': realized_vol(tape, 10.0, w),
        'mean_touch_qty': float(np.nanmean(time_weighted(tape, 'bid_qty', w)['mean'] + time_weighted(tape, 'ask_qty', w)['mean']) / 2),
        'mean_order_size': float(orders['volume'].mean()) if len(orders) else np.nan,
        'mean_trade_size': float(takers['qty'].mean()) if len(takers) else np.nan,
    }


def _durations(tape: Tape, window: tuple[float, float]) -> tuple[pd.DataFrame, np.ndarray]:
    """Quote rows overlapping the window with how long each state lasted inside it."""
    q = tape.quotes
    t = q['t_s'].to_numpy()
    start = max(np.searchsorted(t, window[0], side='right') - 1, 0)
    end = np.searchsorted(t, window[1], side='right')
    rows = q.iloc[start:end]
    times = np.clip(rows['t_s'].to_numpy(), window[0], window[1])
    dur = np.diff(np.append(times, window[1]))
    return rows, dur


def time_weighted(tape: Tape, column: str, window: Window = None) -> dict[str, float]:
    """Time-weighted mean / median / min of a quotes column over its two-sided states."""
    rows, dur = _durations(tape, bounds(tape, window))
    vals = rows[column].to_numpy(dtype='float64')
    ok = ~np.isnan(vals) & (dur > 0) & rows['mid'].notna().to_numpy()
    if not ok.any():
        return {'mean': np.nan, 'median': np.nan, 'min': np.nan, 'share_at_min': np.nan}
    v, d = vals[ok], dur[ok]
    order = np.argsort(v, kind='stable')
    cum = np.cumsum(d[order]) / d.sum()
    vmin = v.min()
    return {
        'mean': float(np.average(v, weights=d)),
        'median': float(v[order][np.searchsorted(cum, 0.5)]),
        'min': float(vmin),
        'share_at_min': float(d[v == vmin].sum() / d.sum()),
    }


def time_weighted_hist(tape: Tape, column: str, window: Window = None) -> pd.Series:
    """Share of (two-sided) time spent at each value of an integer-valued quotes column."""
    rows, dur = _durations(tape, bounds(tape, window))
    ok = rows['mid'].notna().to_numpy() & (dur > 0)
    s = pd.Series(dur[ok]).groupby(rows[column].to_numpy()[ok]).sum()
    return s / s.sum()


def one_sided_share(tape: Tape, window: Window = None) -> float:
    """Share of time with a side of the book empty (ignoring the microsecond cancel/replace gaps)."""
    rows, dur = _durations(tape, bounds(tape, window))
    gap = rows['mid'].isna().to_numpy() & (dur > 0.001)
    return float(dur[gap].sum() / max(dur.sum(), 1e-9))


def activity(tape: Tape, bin_s: float = 60.0, window: Window = None) -> pd.DataFrame:
    """Per time bin: events, inserts, cancels, taker orders, traded volume, mid (last) and spread (time-mean)."""
    w = bounds(tape, window)
    edges = np.arange(w[0], w[1] + bin_s, bin_s)
    ev_t = (tape.events['t_us'].to_numpy() - tape.start_us) / 1e6
    is_ins = (tape.events['op'] == 'I').to_numpy()
    takers = taker_orders(tape, w)
    out = pd.DataFrame(index=pd.Index(edges[:-1], name='t_s'))
    out['events'] = np.histogram(ev_t, edges)[0]
    out['inserts'] = np.histogram(ev_t[is_ins], edges)[0]
    out['cancels'] = out['events'] - out['inserts']
    out['taker_orders'] = np.histogram(takers['t_s'], edges)[0]
    out['volume'] = np.histogram(takers['t_s'], edges, weights=takers['qty'])[0]
    out['mid'] = asof(tape, edges[1:])
    sp = mid_on_grid(tape, min(bin_s / 20, 1.0), w)
    spread = pd.Series(asof(tape, sp.index.to_numpy(), 'spread'), index=sp.index)
    out['spread'] = spread.groupby(np.digitize(spread.index, edges) - 1).mean().reindex(range(len(out))).to_numpy()
    out['vol'] = (
        mid_on_grid(tape, 1.0, w).diff().pow(2).groupby(lambda t: int((t - w[0]) // bin_s)).mean().pow(0.5).reindex(range(len(out))).to_numpy()
    )
    return out


# --------------------------------------------------------------------------- volatility & returns


def returns(tape: Tape, dt_s: float, window: Window = None) -> pd.Series:
    """Mid changes over consecutive `dt_s` steps, in ticks."""
    return mid_on_grid(tape, dt_s, window).diff().dropna()


def realized_vol(tape: Tape, dt_s: float, window: Window = None) -> float:
    """sqrt(mean(dmid^2) / dt): ticks per sqrt-second, the `sigma` of Avellaneda-Stoikov."""
    r = returns(tape, dt_s, window)
    return float(np.sqrt((r**2).mean() / dt_s)) if len(r) else np.nan


def vol_signature(tape: Tape, window: Window = None, dts: tuple[float, ...] = (0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 30, 60)) -> pd.DataFrame:
    """Realised vol per sqrt-second vs. sampling interval; flat for a random walk, rising with momentum."""
    return pd.DataFrame({'dt_s': dts, 'vol': [realized_vol(tape, dt, window) for dt in dts]})


def rolling_vol(tape: Tape, window_s: float = 60.0, dt_s: float = 1.0, window: Window = None) -> pd.Series:
    """Realised vol (ticks/sqrt-s) over a trailing `window_s`, from `dt_s` returns."""
    r = mid_on_grid(tape, dt_s, window).diff()
    n = max(int(round(window_s / dt_s)), 2)
    return np.sqrt(r.pow(2).rolling(n, min_periods=n // 2).mean() / dt_s)


def acf(x: pd.Series | np.ndarray, nlags: int = 30) -> np.ndarray:
    """Sample autocorrelation at lags 1..nlags (NaNs dropped)."""
    v = np.asarray(x, dtype='float64')
    v = v[~np.isnan(v)]
    if len(v) < nlags + 2:
        return np.full(nlags, np.nan)
    v = v - v.mean()
    denom = (v * v).sum()
    if denom == 0:
        return np.zeros(nlags)
    return np.array([(v[:-k] * v[k:]).sum() / denom for k in range(1, nlags + 1)])


def acf_by_lag(
    tape: Tape,
    base_dt_s: float = 0.01,
    max_lag_s: float = 60.0,
    n_lags: int = 40,
    window: Window = None,
) -> pd.DataFrame:
    """Autocorrelation of fine-grained mid returns at log-spaced lags in seconds.

    Returns are mid changes over consecutive `base_dt_s` steps (10 ms by default),
    so the lags run from 10 ms to `max_lag_s` on one axis. Bid-ask bounce
    shows up as negative values at the shortest lags, momentum as positive
    values at second-scale lags. `band` is the 95% white-noise band, +-1.96/sqrt(n).
    """
    r = np.diff(mid_on_grid(tape, base_dt_s, window).to_numpy())
    r = r[~np.isnan(r)]
    max_steps = min(int(round(max_lag_s / base_dt_s)), len(r) // 4)
    if max_steps < 1:
        return pd.DataFrame(columns=['lag_s', 'lag_steps', 'acf', 'band'])
    steps = np.unique(np.round(np.logspace(0, np.log10(max_steps), n_lags)).astype(int))
    r = r - r.mean()
    denom = (r * r).sum()
    values = [(r[:-k] * r[k:]).sum() / denom if denom > 0 else np.nan for k in steps]
    return pd.DataFrame({'lag_s': steps * base_dt_s, 'lag_steps': steps, 'acf': values, 'band': 1.96 / np.sqrt(len(r))})


def mid_change_acf(tape: Tape, nlags: int = 50, window: Window = None) -> tuple[np.ndarray, int]:
    """Autocorrelation of consecutive mid changes (event time, not clock time), and how many changes it used.

    Lag 1 near -1 means every move is undone by the next (bounce); positive means moves repeat.
    """
    sizes = mid_changes(tape, window)['size'].to_numpy(dtype='float64')
    return acf(sizes, nlags), len(sizes)


def variance_ratio(tape: Tape, dt_s: float = 1.0, qs: tuple[int, ...] = (2, 5, 10, 30, 60), window: Window = None) -> pd.DataFrame:
    """VR(q) = Var(q-step return) / (q * Var(1-step return)): >1 trending, <1 mean-reverting."""
    mid = mid_on_grid(tape, dt_s, window).dropna().to_numpy()
    r1 = np.diff(mid)
    base = (r1**2).mean() if len(r1) else np.nan
    rows = []
    for q in qs:
        rq = mid[q:] - mid[:-q]
        rows.append({'q': q, 'horizon_s': q * dt_s, 'vr': (rq**2).mean() / (q * base) if base > 0 and len(rq) else np.nan})
    return pd.DataFrame(rows)


def mid_changes(tape: Tape, window: Window = None) -> pd.DataFrame:
    """Every change of the (two-sided) mid: time, size in ticks, and the wait since the previous change."""
    q = _two_sided(tape)
    q = q[_in(q['t_s'], bounds(tape, window))]
    m = q['mid'].to_numpy()
    changed = np.flatnonzero(np.diff(m) != 0) + 1
    t = q['t_s'].to_numpy()[changed]
    return pd.DataFrame({'t_s': t, 'size': np.diff(m)[changed - 1], 'wait_s': np.diff(t, prepend=np.nan)})


# --------------------------------------------------------------------------- liquidity & book


def _depth_mask(tape: Tape, window: Window) -> np.ndarray:
    return _in((tape.depth_t_us - tape.start_us) / 1e6, bounds(tape, window))


def depth_profile(tape: Tape, window: Window = None, max_ticks: int = 20) -> pd.DataFrame:
    """Mean resting quantity at each distance (ticks) from the same-side touch, per side, from the depth snapshots."""
    m = _depth_mask(tape, window)
    out = {'ticks': np.arange(max_ticks)}
    for name, px, qty, sign in (
        ('bid', tape.depth_bid_px[m], tape.depth_bid_qty[m], 1),
        ('ask', tape.depth_ask_px[m], tape.depth_ask_qty[m], -1),
    ):
        acc = np.zeros(max_ticks)
        if len(px):
            dist = sign * (px[:, :1] - px)
            ok = (px > 0) & (dist < max_ticks) & (px[:, :1] > 0)
            np.add.at(acc, dist[ok], qty[ok])
            acc /= len(px)
        out[name] = acc
    return pd.DataFrame(out)


def depth_heatmap(tape: Tape, window: Window = None, max_cols: int = 600) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(times, prices, qty[price, time]) of resting volume by absolute price, downsampled to `max_cols` snapshots.

    Bid quantity is positive and ask quantity negative, for a diverging colour scale.
    """
    m = np.flatnonzero(_depth_mask(tape, window))
    if len(m) == 0:
        return np.array([]), np.array([]), np.zeros((0, 0))
    m = m[:: max(1, len(m) // max_cols)]
    bp, bq = tape.depth_bid_px[m], tape.depth_bid_qty[m]
    ap, aq = tape.depth_ask_px[m], tape.depth_ask_qty[m]
    present = np.concatenate([bp[bp > 0], ap[ap > 0]])
    if len(present) == 0:
        return np.array([]), np.array([]), np.zeros((0, 0))
    lo, hi = int(present.min()), int(present.max())
    prices = np.arange(lo, hi + 1)
    z = np.zeros((len(prices), len(m)))
    cols = np.broadcast_to(np.arange(len(m))[:, None], bp.shape)
    ok = bp > 0
    np.add.at(z, (bp[ok] - lo, cols[ok]), bq[ok])
    ok = ap > 0
    np.add.at(z, (ap[ok] - lo, cols[ok]), -aq[ok])
    return (tape.depth_t_us[m] - tape.start_us) / 1e6, prices, z


def order_lifetimes(tape: Tape, window: Window = None) -> pd.DataFrame:
    """Resting (non-IOC) orders with lifetime in seconds and how they ended."""
    o = tape.orders
    o = o[_in((o['t_us'] - tape.start_us) / 1e6, bounds(tape, window)) & ~o['ioc'] & o['end_t_us'].notna()]
    return pd.DataFrame({'lifetime_s': (o['end_t_us'] - o['t_us']) / 1e6, 'end': o['end'], 'filled': o['filled'], 'volume': o['volume']})


def orders_in(tape: Tape, window: Window = None) -> pd.DataFrame:
    """Inserts in the window, with `kind`: aggressive (crossed the spread) or passive."""
    o = tape.orders[_in((tape.orders['t_us'] - tape.start_us) / 1e6, bounds(tape, window))].copy()
    o['kind'] = np.where(o['crossed'], 'aggressive', 'passive')
    return o


# --------------------------------------------------------------------------- flow & predictability


def ofi_series(tape: Tape, dt_s: float, window: Window = None) -> pd.DataFrame:
    """Order flow imbalance (Cont, Kukanov & Stoikov 2014) and mid change per `dt_s` bin.

    e_n = q^b_n 1[P^b_n >= P^b_{n-1}] - q^b_{n-1} 1[P^b_n <= P^b_{n-1}]
        - q^a_n 1[P^a_n <= P^a_{n-1}] + q^a_{n-1} 1[P^a_n >= P^a_{n-1}],
    summed over consecutive two-sided quote states inside each bin.
    """
    q = _two_sided(tape)
    b, a = q['bid'].to_numpy(), q['ask'].to_numpy()
    qb, qa = q['bid_qty'].to_numpy(dtype='float64'), q['ask_qty'].to_numpy(dtype='float64')
    e = (
        qb[1:] * (b[1:] >= b[:-1])
        - qb[:-1] * (b[1:] <= b[:-1])
        - qa[1:] * (a[1:] <= a[:-1])
        + qa[:-1] * (a[1:] >= a[:-1])
    )
    t = q['t_s'].to_numpy()[1:]
    edges = grid(tape, dt_s, window)
    ofi = np.histogram(t, edges, weights=e)[0]
    mid = asof(tape, edges)
    dmid = np.diff(mid)
    return pd.DataFrame({'t_s': edges[:-1], 'ofi': ofi, 'dmid': dmid, 'dmid_next': np.append(dmid[1:], np.nan)})


def _ols(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    if len(x) < 3 or x.var() == 0:
        return {'beta': np.nan, 'alpha': np.nan, 'r2': np.nan, 'n': int(len(x))}
    beta, alpha = np.polyfit(x, y, 1)
    resid = y - (alpha + beta * x)
    r2 = 1 - resid.var() / y.var() if y.var() > 0 else np.nan
    return {'beta': float(beta), 'alpha': float(alpha), 'r2': float(r2), 'n': int(len(x))}


def ofi_regression(tape: Tape, dts: tuple[float, ...] = (0.1, 1.0, 5.0, 30.0), window: Window = None) -> pd.DataFrame:
    """Per bin size: dmid ~ OFI in the same bin (price impact) and in the next bin (prediction)."""
    rows = []
    for dt in dts:
        s = ofi_series(tape, dt, window)
        same = _ols(s['ofi'].to_numpy(), s['dmid'].to_numpy())
        nxt = _ols(s['ofi'].to_numpy(), s['dmid_next'].to_numpy())
        rows.append({'dt_s': dt, 'beta': same['beta'], 'r2': same['r2'], 'beta_next': nxt['beta'], 'r2_next': nxt['r2'], 'n': same['n']})
    return pd.DataFrame(rows)


def imbalance_response(
    tape: Tape,
    horizons: tuple[float, ...] = (0.1, 1.0, 5.0, 30.0),
    n_bins: int = 10,
    dt_s: float = 0.1,
    window: Window = None,
    latency_s: float = 0.0,
) -> pd.DataFrame:
    """Mean future mid move (ticks) by queue-imbalance bucket: the empirical microprice adjustment.

    Sampled every `dt_s`; with `latency_s`, the move is measured from t + latency
    (what a strategy reacting to the signal could still capture).
    """
    t = grid(tape, dt_s, window)
    imb = asof(tape, t, 'imbalance')
    mid0 = asof(tape, t + latency_s)
    edges = np.linspace(0, 1, n_bins + 1)
    bucket = np.clip(np.digitize(imb, edges) - 1, 0, n_bins - 1)
    rows = []
    for h in horizons:
        dm = asof(tape, t + latency_s + h) - mid0
        df = pd.DataFrame({'bucket': bucket, 'dm': dm, 'imb': imb}).dropna()
        g = df.groupby('bucket')
        res = pd.DataFrame({'imb_mid': g['imb'].mean(), 'mean_dmid': g['dm'].mean(), 'se': g['dm'].std() / np.sqrt(g.size()), 'n': g.size()})
        res['horizon_s'] = h
        rows.append(res.reset_index())
    return pd.concat(rows, ignore_index=True)


def markouts(tape: Tape, horizons: np.ndarray | None = None, window: Window = None) -> pd.DataFrame:
    """Mean signed mid move after aggressive orders, from the taker's side, at each horizon (ticks).

    `impact`  = sign * (mid(t+h) - mid just before the trade): how far price follows the trade.
    `taker_pnl` = sign * (mid(t+h) - average fill price): what the taker made per unit,
    which is minus what the passive side (where a market maker sits) made.
    """
    if horizons is None:
        horizons = np.array([0, 0.005, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 30, 60])
    tk = taker_orders(tape, window)
    if tk.empty:
        return pd.DataFrame(columns=['horizon_s', 'impact', 'impact_se', 'taker_pnl', 'taker_pnl_se', 'n'])
    tr = trades_s(tape, window)
    vwap = (tr['price'] * tr['qty']).groupby(tr['taker_row']).sum() / tr.groupby('taker_row')['qty'].sum()
    fill = vwap.reindex(tk['taker_row']).to_numpy()
    pre = asof(tape, tk['t_s'].to_numpy(), strict=True)
    sign = tk['sign'].to_numpy()
    rows = []
    for h in horizons:
        after = asof(tape, tk['t_s'].to_numpy() + h)
        imp = sign * (after - pre)
        pnl = sign * (after - fill)
        rows.append(
            {
                'horizon_s': float(h),
                'impact': np.nanmean(imp),
                'impact_se': np.nanstd(imp) / np.sqrt(max(np.isfinite(imp).sum(), 1)),
                'taker_pnl': np.nanmean(pnl),
                'taker_pnl_se': np.nanstd(pnl) / np.sqrt(max(np.isfinite(pnl).sum(), 1)),
                'n': int(np.isfinite(imp).sum()),
            }
        )
    return pd.DataFrame(rows)


def fill_intensity(tape: Tape, window: Window = None, max_ticks: int = 15) -> pd.DataFrame:
    """lambda(delta): rate (per second) of aggressive orders reaching `delta` ticks from the pre-trade mid.

    This is the Avellaneda-Stoikov arrival rate of fills for a quote resting at
    distance delta, ignoring queue position. Reported per side and for both
    together; `analysis.fit.fit_intensity` fits A * exp(-k * delta) to it.
    """
    w = bounds(tape, window)
    duration = max(w[1] - w[0], 1e-9)
    tk = taker_orders(tape, w)
    mid = asof(tape, tk['t_s'].to_numpy(), strict=True)
    reach = tk['sign'].to_numpy() * (tk['worst_price'].to_numpy() - mid)
    deltas = np.arange(0, max_ticks + 0.5, 0.5)
    buy = tk['sign'].to_numpy() > 0
    out = pd.DataFrame({'delta': deltas})
    for name, mask in (('both', np.ones(len(tk), bool)), ('buy', buy), ('sell', ~buy)):
        r = reach[mask]
        r = r[~np.isnan(r)]
        out[name] = [(r >= d).sum() / duration for d in deltas]
    out['both'] /= 2  # per side, comparable with buy / sell
    return out


def trade_sign_acf(tape: Tape, nlags: int = 30, window: Window = None) -> np.ndarray:
    """Autocorrelation of successive aggressive orders' signs: order-flow persistence."""
    return acf(taker_orders(tape, window)['sign'].to_numpy(dtype='float64'), nlags)


def cumulative_flow(tape: Tape, window: Window = None) -> pd.DataFrame:
    """Cumulative signed aggressive volume over time."""
    tk = taker_orders(tape, window)
    return pd.DataFrame({'t_s': tk['t_s'], 'cum_flow': (tk['sign'] * tk['qty']).cumsum()})


