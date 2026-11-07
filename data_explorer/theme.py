"""Colours, figure defaults and small layout helpers shared by every tab (dark only, like explorer/)."""

from __future__ import annotations

import math

import plotly.graph_objects as go
from dash import dcc, html

PAGE = '#0d0d0d'
SURFACE = '#1a1a19'
INK = '#f0efec'
INK_2 = '#c3c2b7'
MUTED = '#898781'
GRID = '#2c2c2a'
AXIS = '#383835'

# Categorical slots 1-3 of the reference palette, dark steps (validated all-pairs on the dark surface).
BLUE, ORANGE, AQUA = '#3987e5', '#d95926', '#199e70'
SERIES = [BLUE, ORANGE, AQUA, '#c98500', '#d55181', '#008300', '#9085e9', '#e66767']
BID, ASK = BLUE, ORANGE
BUY, SELL = BLUE, ORANGE  # aggressor side: a buy lifts the ask, but colour follows the taker's side
NEUTRAL = INK_2
# Diverging blue <-> red with a gray midpoint, for bid (+) / ask (-) volume.
DIVERGING = [[0.0, '#e66767'], [0.5, AXIS], [1.0, BLUE]]

DT_OPTIONS = [0.1, 0.5, 1.0, 5.0, 30.0]


def figure(title: str | None = None, *, x: str | None = None, y: str | None = None, height: int = 300, **layout) -> go.Figure:
    """An empty figure with the shared dark styling, recessive grid and a horizontal legend on top."""
    fig = go.Figure()
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif', size=12, color=INK_2),
        title=dict(text=title, font=dict(size=13, color=INK), x=0.01, xanchor='left') if title else None,
        margin=dict(l=56, r=16, t=40 if title else 16, b=40),
        height=height,
        legend=dict(orientation='h', yanchor='bottom', y=1.0, xanchor='right', x=1, bgcolor='rgba(0,0,0,0)'),
        hoverlabel=dict(bgcolor=SURFACE, bordercolor=AXIS, font=dict(color=INK)),
        uirevision='keep',
        **layout,
    )
    axis = dict(gridcolor=GRID, linecolor=AXIS, zerolinecolor=AXIS, tickfont=dict(color=MUTED), title_font=dict(color=MUTED))
    fig.update_xaxes(title_text=x, **axis)
    fig.update_yaxes(title_text=y, **axis)
    return fig


def graph(fig: go.Figure, gid: str | None = None) -> dcc.Graph:
    kwargs = {'id': gid} if gid else {}
    return dcc.Graph(figure=fig, config={'displaylogo': False, 'scrollZoom': False}, **kwargs)


def panel(content, note: str | None = None, *, wide: bool = False) -> html.Div:
    """A chart card with an optional one-line 'how to read it' note under the chart."""
    children = [content]
    if note:
        children.append(html.P(note, className='note'))
    return html.Div(children, className='panel wide' if wide else 'panel')


def grid(*panels) -> html.Div:
    return html.Div(list(panels), className='grid')


def fmt(value, digits: int = 3) -> str:
    """Compact human formatting: thousands separators, sensible significant digits, '-' for missing."""
    if value is None:
        return '-'
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(v):
        return '-'
    if math.isinf(v):
        return '∞'
    if v.is_integer() and abs(v) >= 1:
        return f'{int(v):,}'
    if abs(v) >= 1000:
        return f'{v:,.0f}'
    return f'{v:.{digits}g}'


def tile(label: str, value, sub: str | None = None) -> html.Div:
    children = [html.Div(label, className='tile-label'), html.Div(value if isinstance(value, str) else fmt(value), className='tile-value')]
    if sub:
        children.append(html.Div(sub, className='tile-sub'))
    return html.Div(children, className='tile')


def section(title: str, *children) -> html.Div:
    return html.Div([html.H3(title, className='section-title'), *children], className='section')


def scatter(n_points: int, **kwargs) -> go.Scatter | go.Scattergl:
    """SVG for small series (works everywhere, crisper), WebGL only when there are too many points for SVG."""
    return go.Scattergl(**kwargs) if n_points > 5000 else go.Scatter(**kwargs)


def log_hist(fig: go.Figure, values, name: str, color: str, edges=None, bins: int = 40) -> None:
    """Histogram on a log x-axis, drawn as a filled step line (bars come out hairline-thin on log axes)."""
    import numpy as np

    v = np.asarray(values, dtype='float64')
    v = v[np.isfinite(v) & (v > 0)]
    if not len(v):
        return
    if edges is None:
        edges = np.logspace(np.log10(v.min()), np.log10(v.max() * 1.0001), bins + 1)
    counts, _ = np.histogram(v, edges)
    x = np.repeat(edges, 2)[1:-1]
    y = np.repeat(counts, 2)
    fig.add_trace(
        go.Scatter(x=x, y=y, name=name, mode='lines', line=dict(color=color, width=2), fill='tozeroy', opacity=0.85, hovertemplate=name + ' %{x:.3g}: %{y}<extra></extra>')
    )
    fig.update_xaxes(type='log')


def empty(fig: go.Figure, message: str) -> go.Figure:
    """Say why a chart has nothing to show instead of leaving an empty frame."""
    fig.add_annotation(text=message, x=0.5, y=0.5, xref='paper', yref='paper', showarrow=False, font=dict(color=MUTED, size=13))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig
