"""Volatility & returns: rolling vol, signature plot, return distribution, autocorrelation, variance ratio."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from dash import html
from scipy import stats

from analysis import features as F
from analysis.replay import Tape
from data_explorer import theme as T

FINE_DT_S = 0.01  # base step of the fine autocorrelation chart
MID_DT_S = 0.1  # step of the 100 ms autocorrelation chart
MID_LAGS = 100  # its lags: 0.1 s to 10 s


def render(tape: Tape, window: F.Window, dt: float) -> html.Div:
    t0, t1 = F.bounds(tape, window)
    roll_s = max(10.0, round((t1 - t0) / 60 / 10) * 10)
    rv = F.rolling_vol(tape, roll_s, dt_s=max(dt, 0.1), window=window)
    roll = T.figure(f'Rolling σ over {roll_s:g} s', x='time (s)', y='ticks / √s', showlegend=False, hovermode='x unified')
    roll.add_trace(T.scatter(len(rv.index), x=rv.index, y=rv.values, name='σ', mode='lines', line=dict(color=T.AQUA, width=2)))

    sig = F.vol_signature(tape, window)
    sigf = T.figure('Volatility signature', x='sampling interval (s)', y='σ (ticks / √s)', showlegend=False)
    sigf.add_trace(go.Scatter(x=sig['dt_s'], y=sig['vol'], mode='lines+markers', line=dict(color=T.BLUE, width=2), marker=dict(size=8), hovertemplate='%{x}s: σ %{y:.3f}<extra></extra>'))
    sigf.update_xaxes(type='log')

    r = F.returns(tape, dt, window).to_numpy()
    rf = T.figure(f'{dt:g} s mid returns', x='Δmid (ticks)', y='share of intervals', showlegend=False)
    sub = ''
    if len(r):
        lo, hi = np.percentile(r, [0.5, 99.5])
        lo, hi = np.floor(min(lo, -1)), np.ceil(max(hi, 1))
        edges = np.arange(lo - 0.25, hi + 0.75, 0.5)
        counts, _ = np.histogram(r, edges)
        centers = (edges[:-1] + edges[1:]) / 2
        rf.add_trace(go.Bar(x=centers, y=counts / len(r), name='observed', marker=dict(color=T.BLUE), hovertemplate='%{x}: %{y:.2%}<extra></extra>'))
        sd = r.std()
        if sd > 0:
            xs = np.linspace(lo, hi, 200)
            rf.add_trace(go.Scatter(x=xs, y=stats.norm.pdf(xs, r.mean(), sd) * 0.5, name='normal, same σ', mode='lines', line=dict(color=T.INK_2, width=2, dash='dot')))
        sub = f'excess kurtosis {stats.kurtosis(r):.2f}, skew {stats.skew(r):.2f}, {np.mean(r == 0):.0%} of intervals unchanged, n = {len(r):,}'
        rf.update_yaxes(tickformat='.0%')

    nlags = 30
    ac = F.acf(F.returns(tape, dt, window), nlags)
    band = 1.96 / np.sqrt(max(len(r), 1))
    acff = T.figure(f'Autocorrelation of {dt:g} s returns', x='lag (intervals)', y='correlation', showlegend=False)
    acff.add_hrect(y0=-band, y1=band, fillcolor=T.AXIS, opacity=0.6, line_width=0, layer='below')
    acff.add_trace(go.Bar(x=np.arange(1, nlags + 1), y=ac, marker=dict(color=np.where(ac >= 0, T.BLUE, T.ORANGE)), hovertemplate='lag %{x}: %{y:.3f}<extra></extra>'))

    fine = F.acf_by_lag(tape, FINE_DT_S, window=window)
    finef = T.figure(f'Autocorrelation of {FINE_DT_S * 1000:g} ms returns by lag', x='lag (s, log)', y='correlation', showlegend=False, hovermode='x unified')
    if len(fine):
        fb = float(fine['band'].iloc[0])
        finef.add_hrect(y0=-fb, y1=fb, fillcolor=T.AXIS, opacity=0.6, line_width=0, layer='below')
        finef.add_hline(y=0, line=dict(color=T.MUTED, width=1))
        finef.add_trace(
            go.Scatter(
                x=fine['lag_s'], y=fine['acf'], name='ACF', mode='lines+markers',
                line=dict(color=T.BLUE, width=2), marker=dict(size=8, color=np.where(fine['acf'] >= 0, T.BLUE, T.ORANGE), line=dict(color=T.SURFACE, width=2)),
                hovertemplate='lag %{x:.3g}s: %{y:.4f}<extra></extra>',
            )
        )
        finef.add_vline(x=0.005 if FINE_DT_S <= 0.005 else FINE_DT_S, line=dict(color=T.MUTED, dash='dash'))
        finef.update_xaxes(type='log')
    else:
        T.empty(finef, 'Window too short')

    r_mid = F.returns(tape, MID_DT_S, window)
    ac_mid = F.acf(r_mid, MID_LAGS)
    mid_band = 1.96 / np.sqrt(max(len(r_mid), 1))
    midf = T.figure(f'Autocorrelation of {MID_DT_S * 1000:g} ms returns, every lag to {MID_DT_S * MID_LAGS:g} s', x='lag (s)', y='correlation', showlegend=False)
    if np.isfinite(ac_mid).any():
        midf.add_hrect(y0=-mid_band, y1=mid_band, fillcolor=T.AXIS, opacity=0.6, line_width=0, layer='below')
        midf.add_trace(
            go.Bar(
                x=np.arange(1, MID_LAGS + 1) * MID_DT_S, y=ac_mid,
                marker=dict(color=np.where(ac_mid >= 0, T.BLUE, T.ORANGE)),
                hovertemplate='lag %{x:.1f}s: %{y:.4f}<extra></extra>',
            )
        )
        midf.update_xaxes(dtick=1)
    else:
        T.empty(midf, f'Window too short for {MID_LAGS} lags of {MID_DT_S * 1000:g} ms')

    nl_ev = 50
    ev_ac, n_changes = F.mid_change_acf(tape, nl_ev, window)
    ev_band = 1.96 / np.sqrt(max(n_changes, 1))
    evf = T.figure('Autocorrelation of consecutive mid changes (tick by tick)', x='lag (mid changes)', y='correlation', showlegend=False)
    evf.add_hrect(y0=-ev_band, y1=ev_band, fillcolor=T.AXIS, opacity=0.6, line_width=0, layer='below')
    evf.add_trace(go.Bar(x=np.arange(1, nl_ev + 1), y=ev_ac, marker=dict(color=np.where(ev_ac >= 0, T.BLUE, T.ORANGE)), hovertemplate='lag %{x}: %{y:.3f}<extra></extra>'))

    vr = F.variance_ratio(tape, dt_s=max(dt, 0.1), window=window)
    vrf = T.figure('Variance ratio', x='horizon (s)', y='VR(q)', showlegend=False)
    vrf.add_hline(y=1, line=dict(color=T.MUTED, dash='dot'))
    vrf.add_trace(go.Scatter(x=vr['horizon_s'], y=vr['vr'], mode='lines+markers', line=dict(color=T.AQUA, width=2), marker=dict(size=8), hovertemplate='%{x}s: %{y:.2f}<extra></extra>'))
    vrf.update_xaxes(type='log')

    mc = F.mid_changes(tape, window)
    waits = mc['wait_s'].dropna().to_numpy()
    waitf = T.figure('Time between mid changes', x='seconds (log)', y='count', showlegend=False)
    T.log_hist(waitf, waits, 'mid changes', T.BLUE)
    jumps = mc['size'].round(1).value_counts().sort_index()
    jumpf = T.figure('Size of mid changes', x='Δmid (ticks)', y='count', showlegend=False)
    jumpf.add_trace(go.Bar(x=jumps.index, y=jumps.values, marker=dict(color=T.AQUA)))

    return T.grid(
        T.panel(T.graph(roll), 'Clustering (calm and wild stretches) means a fixed σ under- and over-quotes in turn: estimate it online.', wide=True),
        T.panel(T.graph(sigf), 'Flat: random walk. Rising with the interval: momentum. Falling: bid-ask bounce or mean reversion. Pick σ at the horizon you hold inventory over.'),
        T.panel(T.graph(rf), sub),
        T.panel(T.graph(acff), 'Bars outside the gray band are significant. Positive at short lags: trends (take); negative: reversion (make).'),
        T.panel(T.graph(vrf), 'Above 1: moves persist over that horizon. Below 1: they revert.'),
        T.panel(
            T.graph(finef),
            'One axis from 10 ms to 60 s, so it does not depend on the sampling selector. Negative at the shortest lags: '
            'bid-ask bounce or quote flicker. Positive at second-scale lags: momentum. Dashed line: our 5 ms latency. '
            'Values look small because most 10 ms returns are zero: compare them with the gray band, not with 1.',
            wide=True,
        ),
        T.panel(
            T.graph(midf),
            f'{len(r_mid):,} returns. Between the two charts above: the lag at which bars turn from blue to orange is how long a move keeps going '
            'before it starts to revert, which is a natural holding time or signal horizon.',
            wide=True,
        ),
        T.panel(T.graph(evf), f'{n_changes:,} mid changes. Lag 1 strongly negative: moves get undone (fade them). Positive: moves continue (follow them).'),
        T.panel(T.graph(waitf), f'{len(mc):,} mid changes. Changes are the events a quoting strategy has to react to.'),
        T.panel(T.graph(jumpf), 'Mostly ±0.5 / ±1 means the price moves tick by tick; large jumps mean gaps in the book.'),
    )
