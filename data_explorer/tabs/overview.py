"""Overview: headline numbers, fitted strategy parameters and per-minute activity."""

from __future__ import annotations

import json

import plotly.graph_objects as go
from dash import html
from plotly.subplots import make_subplots

from analysis import features as F
from analysis.fit import fit_all
from analysis.replay import Tape
from data_explorer import theme as T


def render(tape: Tape, window: F.Window, dt: float) -> html.Div:
    s = F.summary(tape, window)
    p = fit_all(tape, window)
    one_tick = f"{s['spread_1tick_share']:.0%} of time at 1 tick" if s['spread_min'] == 1 else f"min {T.fmt(s['spread_min'])} ticks"

    tiles = html.Div(
        [
            T.tile('Window', f"{s['duration_s'] / 60:.1f} min", f"{T.fmt(s['events'])} events"),
            T.tile('Events / s', s['events_per_s'], f"cancel/insert {s['cancel_to_insert']:.2f}"),
            T.tile('Aggressive orders / min', s['trades_per_min'], f"{T.fmt(s['trades'])} fills"),
            T.tile('Traded volume', s['traded_volume'], f"{s['buy_volume_share']:.0%} buyer-initiated"),
            T.tile('Mid', f"{T.fmt(s['mid_start'])} → {T.fmt(s['mid_end'])}", f"range {T.fmt(s['price_min'])}–{T.fmt(s['price_max'])}"),
            T.tile('Spread (time-weighted)', s['spread_mean'], one_tick),
            T.tile('One-sided book', f"{s['one_sided_share']:.1%}", 'of time a side is empty'),
            T.tile('σ at 100 ms', s['vol_100ms'], 'ticks / √s'),
            T.tile('σ at 1 s', s['vol_1s'], f"10 s: {T.fmt(s['vol_10s'])}"),
            T.tile('Touch size', s['mean_touch_qty'], 'mean qty at best, per side'),
            T.tile('Order / fill size', f"{T.fmt(s['mean_order_size'])} / {T.fmt(s['mean_trade_size'])}", 'mean insert / aggressive order'),
            T.tile('Passive fill rate', f"{s['passive_fill_rate']:.1%}", 'resting orders that got any fill'),
        ],
        className='tiles',
    )

    snippet = '\n'.join(
        [
            f"# {tape.name}, window {p['window_s'][0]:.0f}-{p['window_s'][1]:.0f} s  (python -m analysis.fit)",
            f"SIGMA_INIT = {T.fmt(p['sigma_100ms'], 4)}   # ticks/sqrt(s) at 100 ms; 1 s: {T.fmt(p['sigma_1s'], 4)}",
            f"K_INIT = {T.fmt(p['k_depth'], 4)}       # ln(1 + 1/mean sweep depth), depth {T.fmt(p['mean_sweep_depth'], 3)}",
            f"# intensity fit: A = {T.fmt(p['A'], 4)} /s, k = {T.fmt(p['k_intensity'], 4)} /tick ({p['fit_points']} points)",
            f"# OFI beta at 1 s = {T.fmt(p['ofi_beta_1s'], 4)} ticks/lot (R² {T.fmt(p['ofi_r2_1s'], 3)}), next-bin R² {T.fmt(p['ofi_r2_next_1s'], 3)}",
            f"# taker markout vs fill: 1 s {T.fmt(p['taker_pnl_1s'], 3)}, 10 s {T.fmt(p['taker_pnl_10s'], 3)} ticks",
            f"IMBALANCE_DMID_1S = {json.dumps(p['imbalance_dmid_1s'])}",
        ]
    )

    act = F.activity(tape, bin_s=_bin(tape, window), window=window)
    rows = [('events', 'Events', T.NEUTRAL), ('taker_orders', 'Aggressive orders', T.BLUE), ('spread', 'Spread (ticks)', T.ORANGE), ('vol', 'σ 1 s (ticks/√s)', T.AQUA)]
    fig = make_subplots(rows=len(rows), cols=1, shared_xaxes=True, vertical_spacing=0.04, subplot_titles=[r[1] for r in rows])
    for i, (col, name, color) in enumerate(rows, start=1):
        fig.add_trace(go.Scatter(x=act.index, y=act[col], name=name, mode='lines', line=dict(color=color, width=2, shape='hv')), row=i, col=1)
    base = T.figure(height=560, showlegend=False)
    fig.update_layout(base.layout)
    fig.update_layout(margin=dict(l=56, r=16, t=28, b=40))
    fig.update_annotations(font=dict(size=12, color=T.INK_2), x=0.0, xanchor='left')
    fig.update_xaxes(gridcolor=T.GRID, linecolor=T.AXIS, tickfont=dict(color=T.MUTED))
    fig.update_yaxes(gridcolor=T.GRID, linecolor=T.AXIS, tickfont=dict(color=T.MUTED))
    fig.update_xaxes(title_text='time (s)', row=len(rows), col=1)

    return html.Div(
        [
            T.section('Tape fingerprint', tiles),
            T.section(
                'Fitted parameters (selected window)',
                html.Div([html.Div(snippet, className='code'), html.P('Copy into a strategy: submissions cannot read files.', className='note')], className='panel'),
            ),
            T.section(
                f'Activity per {_bin(tape, window):g} s',
                T.panel(T.graph(fig), 'Regime changes show up here first: bursts of activity, spread widening, volatility clustering.', wide=True),
            ),
        ]
    )


def _bin(tape: Tape, window: F.Window) -> float:
    t0, t1 = F.bounds(tape, window)
    span = t1 - t0
    for b in (1, 5, 10, 30, 60, 120, 300):
        if span / b <= 120:
            return float(b)
    return 600.0
