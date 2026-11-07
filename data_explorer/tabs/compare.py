"""Round comparison: every loaded tape side by side (whole tapes, the window is ignored)."""

from __future__ import annotations

import plotly.graph_objects as go
from dash import html

from analysis import features as F
from analysis.fit import fit_all
from analysis.replay import Tape
from data_explorer import theme as T

ROWS = [
    ('duration_s', 'Duration (s)'),
    ('events_per_s', 'Events / s'),
    ('cancel_to_insert', 'Cancel / insert'),
    ('trades_per_min', 'Aggressive orders / min'),
    ('traded_volume', 'Traded volume'),
    ('buy_volume_share', 'Buyer-initiated share'),
    ('mid_start', 'Mid start'),
    ('mid_end', 'Mid end'),
    ('spread_mean', 'Spread mean (ticks)'),
    ('spread_min', 'Spread min'),
    ('one_sided_share', 'One-sided share'),
    ('vol_100ms', 'σ 100 ms'),
    ('vol_1s', 'σ 1 s'),
    ('vol_10s', 'σ 10 s'),
    ('mean_touch_qty', 'Touch size'),
    ('mean_order_size', 'Insert size'),
    ('mean_trade_size', 'Aggressive order size'),
    ('passive_fill_rate', 'Passive fill rate'),
]
FIT_ROWS = [('k_depth', 'k (sweep depth)'), ('A', 'A (intensity)'), ('k_intensity', 'k (intensity)'), ('ofi_r2_next_1s', 'OFI next-bin R² 1 s'), ('taker_pnl_1s', 'Taker markout 1 s'), ('taker_pnl_10s', 'Taker markout 10 s')]


def render(tapes: list[Tape]) -> html.Div:
    if len(tapes) < 2:
        return html.P('Load two or more tapes to compare them:  data-explorer --data data/round1.csv data/round2.csv ...', className='note')
    summaries = [F.summary(t) for t in tapes]
    fits = [fit_all(t) for t in tapes]
    header = html.Tr([html.Th('')] + [html.Th(t.name) for t in tapes])
    body = [html.Tr([html.Td(label)] + [html.Td(T.fmt(s[key])) for s in summaries]) for key, label in ROWS]
    body += [html.Tr([html.Td(label)] + [html.Td(T.fmt(f[key])) for f in fits]) for key, label in FIT_ROWS]
    table = html.Table([header, *body], className='stats')

    colors = dict(zip([t.name for t in tapes], T.SERIES))
    sig = T.figure('Volatility signature', x='sampling interval (s)', y='σ (ticks / √s)')
    spread = T.figure('Spread distribution', x='spread (ticks)', y='share of time', barmode='group')
    ret = T.figure('1 s returns', x='Δmid (ticks)', y='share of intervals')
    imp = T.figure('Price impact after aggressive orders', x='seconds after (log)', y='ticks')
    depth = T.figure('Depth profile (bid + ask) / 2', x='ticks behind best', y='mean qty')
    for t in tapes:
        c = colors[t.name]
        s = F.vol_signature(t)
        sig.add_trace(go.Scatter(x=s['dt_s'], y=s['vol'], name=t.name, mode='lines+markers', line=dict(color=c, width=2), marker=dict(size=8)))
        h = F.time_weighted_hist(t, 'spread')
        spread.add_trace(go.Bar(x=h.index, y=h.values, name=t.name, marker=dict(color=c)))
        r = F.returns(t, 1.0)
        vc = r.round().clip(-15, 15).value_counts(normalize=True).sort_index()
        ret.add_trace(go.Scatter(x=vc.index, y=vc.values, name=t.name, mode='lines+markers', line=dict(color=c, width=2), marker=dict(size=8)))
        m = F.markouts(t)
        m = m[m['horizon_s'] > 0]
        imp.add_trace(go.Scatter(x=m['horizon_s'], y=m['impact'], name=t.name, mode='lines+markers', line=dict(color=c, width=2), marker=dict(size=8)))
        d = F.depth_profile(t)
        depth.add_trace(go.Scatter(x=d['ticks'], y=(d['bid'] + d['ask']) / 2, name=t.name, mode='lines+markers', line=dict(color=c, width=2), marker=dict(size=8)))
    sig.update_xaxes(type='log')
    imp.update_xaxes(type='log')
    spread.update_yaxes(tickformat='.0%')
    ret.update_yaxes(type='log')

    return html.Div(
        [
            T.section('Side by side (whole tapes)', html.Div(table, className='panel', style={'padding': '8px', 'overflowX': 'auto'})),
            T.grid(T.panel(T.graph(sig)), T.panel(T.graph(spread)), T.panel(T.graph(ret), 'Log scale: fat tails stand out.'), T.panel(T.graph(imp)), T.panel(T.graph(depth))),
        ]
    )


