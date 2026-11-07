"""Predictability & adverse selection: imbalance -> future mid, OFI, markouts, and what survives 5 ms latency."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from dash import html

from analysis import features as F
from analysis.replay import Tape
from data_explorer import theme as T

LATENCY_S = 0.005
HORIZONS = (0.1, 1.0, 5.0, 30.0)
MIN_BUCKET = 30  # imbalance buckets with fewer samples are too noisy to plot


def render(tape: Tape, window: F.Window, dt: float) -> html.Div:
    resp = F.imbalance_response(tape, HORIZONS, window=window)
    imb = T.figure('Future mid move by queue imbalance', x='imbalance  Qbid / (Qbid + Qask)', y='mean Δmid (ticks)', hovermode='x unified')
    imb.add_hline(y=0, line=dict(color=T.MUTED, dash='dot'))
    for color, h in zip(T.SERIES, HORIZONS):
        r = resp[(resp['horizon_s'] == h) & (resp['n'] >= MIN_BUCKET)]
        imb.add_trace(go.Scatter(x=r['imb_mid'], y=r['mean_dmid'], name=f'{h:g} s', mode='lines+markers', line=dict(color=color, width=2), marker=dict(size=8), error_y=dict(array=1.96 * r['se'], color=color, thickness=1)))

    lat = T.figure('Is the imbalance edge still there 5 ms later?', x='imbalance', y='mean Δmid over the next 100 ms (ticks)', hovermode='x unified')
    lat.add_hline(y=0, line=dict(color=T.MUTED, dash='dot'))
    for color, delay, name in ((T.BLUE, 0.0, 'measured from now'), (T.ORANGE, LATENCY_S, 'measured from now + 5 ms')):
        r = F.imbalance_response(tape, (0.1,), window=window, latency_s=delay)
        r = r[r['n'] >= MIN_BUCKET]
        lat.add_trace(go.Scatter(x=r['imb_mid'], y=r['mean_dmid'], name=name, mode='lines+markers', line=dict(color=color, width=2), marker=dict(size=8)))

    reg = F.ofi_regression(tape, (0.1, 0.5, 1.0, 5.0, 30.0), window)
    ofit = T.figure('Order flow imbalance → mid move (R²)', x='bin size (s)', y='R²')
    ofit.add_trace(go.Scatter(x=reg['dt_s'], y=reg['r2'], name='same bin (impact)', mode='lines+markers', line=dict(color=T.BLUE, width=2), marker=dict(size=8)))
    ofit.add_trace(go.Scatter(x=reg['dt_s'], y=reg['r2_next'], name='next bin (prediction)', mode='lines+markers', line=dict(color=T.ORANGE, width=2), marker=dict(size=8)))
    ofit.update_xaxes(type='log')

    s = F.ofi_series(tape, dt, window).dropna()
    if len(s) > 5000:
        s = s.sample(5000, random_state=0)
    sc = T.figure(f'OFI vs Δmid, {dt:g} s bins', x='OFI (lots)', y='Δmid (ticks)', showlegend=False)
    sc.add_trace(T.scatter(len(s['ofi']), x=s['ofi'], y=s['dmid'], mode='markers', marker=dict(color=T.BLUE, size=8, opacity=0.35, line=dict(color=T.SURFACE, width=1)), hovertemplate='OFI %{x}  Δmid %{y}<extra></extra>'))
    r = reg.set_index('dt_s')
    beta = r['beta'].get(dt, np.nan)
    if np.isfinite(beta) and len(s):
        xs = np.linspace(s['ofi'].min(), s['ofi'].max(), 2)
        sc.add_trace(go.Scatter(x=xs, y=beta * xs, mode='lines', line=dict(color=T.INK_2, dash='dot', width=2), name='fit'))

    mk = F.markouts(tape, window=window)
    mkf = T.figure('After an aggressive order: mid move in its direction', x='seconds after (log)', y='ticks', hovermode='x unified')
    mkf.add_hline(y=0, line=dict(color=T.MUTED, dash='dot'))
    mk_pos = mk[mk['horizon_s'] > 0]
    mkf.add_trace(go.Scatter(x=mk_pos['horizon_s'], y=mk_pos['impact'], name='price impact (vs pre-trade mid)', mode='lines+markers', line=dict(color=T.BLUE, width=2), marker=dict(size=8), error_y=dict(array=1.96 * mk_pos['impact_se'], color=T.BLUE, thickness=1)))
    mkf.add_trace(go.Scatter(x=mk_pos['horizon_s'], y=mk_pos['taker_pnl'], name='taker PnL per lot (vs fill) = −maker PnL', mode='lines+markers', line=dict(color=T.ORANGE, width=2), marker=dict(size=8)))
    mkf.add_vline(x=LATENCY_S, line=dict(color=T.MUTED, dash='dash'), annotation_text='5 ms', annotation_font_color=T.MUTED)
    mkf.update_xaxes(type='log')

    table = html.Table(
        [html.Tr([html.Th('bin'), html.Th('β (ticks/lot)'), html.Th('R² same'), html.Th('β next'), html.Th('R² next'), html.Th('n')])]
        + [html.Tr([html.Td(f'{row.dt_s:g} s'), html.Td(T.fmt(row.beta, 3)), html.Td(T.fmt(row.r2, 3)), html.Td(T.fmt(row.beta_next, 3)), html.Td(T.fmt(row.r2_next, 3)), html.Td(T.fmt(row.n))]) for row in reg.itertuples()],
        className='stats',
    )

    return T.grid(
        T.panel(T.graph(imb), f'Buckets with under {MIN_BUCKET} samples hidden. A monotone S-curve means imbalance predicts the next move: use the microprice instead of the mid, or lean quotes with it.'),
        T.panel(T.graph(lat), 'Our orders land 5 ms after we decide. If the two curves match, the signal is still worth trading on.'),
        T.panel(T.graph(mkf), 'Positive orange line: passive fills lose money at that horizon (adverse selection). The quoted half-spread must cover it.', wide=True),
        T.panel(T.graph(ofit), 'Same-bin R² is price impact (descriptive). Next-bin R² is real predictability.'),
        T.panel(T.graph(sc), 'Change the bin size with the sampling selector at the top.'),
        T.panel(html.Div([html.H3('OFI regression  Δmid = β·OFI', className='section-title'), table], style={'padding': '8px'})),
    )
