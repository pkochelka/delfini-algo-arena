"""Trades & order flow: intensity, arrival times, sizes, sweep depth, fill intensity, flow persistence."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from dash import html
from plotly.subplots import make_subplots

from analysis import features as F
from analysis.fit import fit_intensity, fit_k_depth
from analysis.replay import Tape
from data_explorer import theme as T


def render(tape: Tape, window: F.Window, dt: float) -> html.Div:
    t0, t1 = F.bounds(tape, window)
    tk = F.taker_orders(tape, window)
    if tk.empty:
        return html.Div('No aggressive orders in this window.', className='note')

    bin_s = max(1.0, round((t1 - t0) / 120))
    edges = np.arange(t0, t1 + bin_s, bin_s)
    rate = T.figure(f'Aggressive orders per second ({bin_s:g} s bins)', x='time (s)', y='orders / s', barmode='stack', hovermode='x unified')
    for sign, name, color in ((1, 'Buy', T.BUY), (-1, 'Sell', T.SELL)):
        c, _ = np.histogram(tk.loc[tk['sign'] == sign, 't_s'], edges)
        rate.add_trace(go.Bar(x=edges[:-1], y=c / bin_s, name=name, marker=dict(color=color)))

    gaps = np.diff(tk['t_s'].to_numpy())
    gapf = T.figure('Time between aggressive orders', x='seconds', y='count (log)')
    sub_gap = ''
    if len(gaps) > 1:
        hi = np.percentile(gaps, 99)
        counts, e = np.histogram(gaps[gaps <= hi], 50)
        centers = (e[:-1] + e[1:]) / 2
        gapf.add_trace(go.Bar(x=centers, y=counts, name='observed', marker=dict(color=T.BLUE)))
        lam = 1 / gaps.mean()
        gapf.add_trace(go.Scatter(x=centers, y=len(gaps) * (np.exp(-lam * e[:-1]) - np.exp(-lam * e[1:])), name='Poisson, same rate', mode='lines', line=dict(color=T.INK_2, dash='dot', width=2)))
        gapf.update_yaxes(type='log')
        cv = gaps.std() / gaps.mean()
        sub_gap = f'Coefficient of variation {cv:.2f} (1 for Poisson; above 1 means clustered arrivals, below 1 means regular ones).'

    sizef = T.figure('Aggressive order size', x='quantity', y='count', barmode='overlay')
    hi = max(int(np.percentile(tk['qty'], 99.5)), 1)
    for sign, name, color in ((1, 'Buy', T.BUY), (-1, 'Sell', T.SELL)):
        sizef.add_trace(go.Histogram(x=tk.loc[tk['sign'] == sign, 'qty'], name=name, marker=dict(color=color), opacity=0.7, xbins=dict(start=0.5, end=hi + 0.5, size=max(1, hi // 50))))

    kd = fit_k_depth(tape, window)
    dep = tk['depth'].value_counts().sort_index()
    depf = T.figure('Levels swept past the touch', x='ticks past the best opposite price', y='share of orders')
    depf.add_trace(go.Bar(x=dep.index, y=dep.values / dep.sum(), name='observed', marker=dict(color=T.AQUA)))
    if np.isfinite(kd['k_depth']):
        n = np.arange(0, dep.index.max() + 1)
        k = kd['k_depth']
        depf.add_trace(go.Scatter(x=n, y=np.exp(-k * n) - np.exp(-k * (n + 1)), name=f'geometric, k = {k:.2f}', mode='lines+markers', line=dict(color=T.INK_2, dash='dot', width=2)))
    depf.update_yaxes(tickformat='.0%')
    depf.update_xaxes(dtick=1)

    lam = F.fill_intensity(tape, window)
    fit = fit_intensity(tape, window)
    lamf = T.figure('Fill intensity λ(δ) for a quote δ ticks from the mid', x='δ (ticks from mid)', y='hits per second (log)')
    for col, name, color in (('buy', 'Ask side hit by buys', T.BUY), ('sell', 'Bid side hit by sells', T.SELL)):
        m = lam[col] > 0
        lamf.add_trace(go.Scatter(x=lam.loc[m, 'delta'], y=lam.loc[m, col], name=name, mode='lines+markers', line=dict(color=color, width=2, shape='hv'), marker=dict(size=8)))
    if not (lam[['buy', 'sell']] > 0).to_numpy().any():
        T.empty(lamf, 'No aggressive orders')
    if np.isfinite(fit['A']):
        xs = lam['delta']
        lamf.add_trace(go.Scatter(x=xs, y=fit['A'] * np.exp(-fit['k_intensity'] * xs), name=f"A·e^(−kδ): A={fit['A']:.3g}, k={fit['k_intensity']:.3g}", mode='lines', line=dict(color=T.INK_2, dash='dot', width=2)))
    lamf.update_yaxes(type='log')

    cf = F.cumulative_flow(tape, window)
    mid = F.mid_on_grid(tape, max((t1 - t0) / 2000, 0.1), window)
    flow = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, subplot_titles=['Cumulative signed aggressive volume', 'Mid'])
    flow.add_trace(T.scatter(len(cf['t_s']), x=cf['t_s'], y=cf['cum_flow'], name='Net buy volume', mode='lines', line=dict(color=T.AQUA, width=2, shape='hv')), row=1, col=1)
    flow.add_trace(T.scatter(len(mid.index), x=mid.index, y=mid.values, name='Mid', mode='lines', line=dict(color=T.NEUTRAL, width=1.5)), row=2, col=1)
    flow.update_layout(T.figure(height=420, showlegend=False, hovermode='x unified').layout)
    flow.update_annotations(font=dict(size=12, color=T.INK_2), x=0.0, xanchor='left')
    flow.update_xaxes(gridcolor=T.GRID, tickfont=dict(color=T.MUTED))
    flow.update_yaxes(gridcolor=T.GRID, tickfont=dict(color=T.MUTED))

    nl = 30
    sa = F.trade_sign_acf(tape, nl, window)
    band = 1.96 / np.sqrt(len(tk))
    sacf = T.figure('Autocorrelation of aggressive order signs', x='lag (orders)', y='correlation', showlegend=False)
    sacf.add_hrect(y0=-band, y1=band, fillcolor=T.AXIS, opacity=0.6, line_width=0, layer='below')
    sacf.add_trace(go.Bar(x=np.arange(1, nl + 1), y=sa, marker=dict(color=np.where(sa >= 0, T.BLUE, T.ORANGE))))

    return T.grid(
        T.panel(T.graph(rate), f'{len(tk):,} aggressive orders, {tk["qty"].sum():,} lots. Fills for a passive quote only come from these.', wide=True),
        T.panel(T.graph(gapf), sub_gap),
        T.panel(T.graph(sizef), 'Large aggressive orders are the ones that sweep through our quote and leave us adversely selected.'),
        T.panel(T.graph(depf), f"Mean depth {kd['mean_sweep_depth']:.3g} → k = ln(1 + 1/mean) = {kd['k_depth']:.3g}: the estimator avellaneda_stoikov.py uses online."),
        T.panel(T.graph(lamf), 'The Avellaneda-Stoikov arrival model. A straight line on this log axis means the exponential model fits; a kink means it does not.'),
        T.panel(T.graph(flow), 'If price follows net flow closely, aggressive orders carry information.', wide=True),
        T.panel(T.graph(sacf), 'Positive: buys follow buys (split metaorders). Their signs then predict the next order.'),
    )
