"""Price & spread: quotes with trades, spread over time and its distribution, depth heatmap."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from dash import html

from analysis import features as F
from analysis.replay import Tape
from data_explorer import theme as T

MAX_QUOTE_POINTS = 20_000
MAX_TRADE_POINTS = 30_000


def quotes_in(tape: Tape, window: F.Window):
    """Two-sided quote states in the window, or the as-of states on a grid when there are too many to draw."""
    t0, t1 = F.bounds(tape, window)
    q = tape.quotes
    q = q[(q['t_s'] >= t0) & (q['t_s'] <= t1) & q['mid'].notna()]
    if len(q) <= MAX_QUOTE_POINTS:
        return q['t_s'].to_numpy(), {c: q[c].to_numpy() for c in ('bid', 'ask', 'microprice', 'spread')}, False
    t = np.linspace(t0, t1, MAX_QUOTE_POINTS // 4)
    return t, {c: F.asof(tape, t, c) for c in ('bid', 'ask', 'microprice', 'spread')}, True


def render(tape: Tape, window: F.Window, dt: float) -> html.Div:
    t, q, sampled = quotes_in(tape, window)

    price = T.figure('Best bid / ask, microprice and aggressive orders', x='time (s)', y='price (ticks)', height=460, hovermode='x unified')
    price.add_trace(T.scatter(len(t), x=t, y=q['bid'], name='Best bid', mode='lines', line=dict(color=T.BID, width=1.5, shape='hv')))
    price.add_trace(T.scatter(len(t), x=t, y=q['ask'], name='Best ask', mode='lines', line=dict(color=T.ASK, width=1.5, shape='hv')))
    price.add_trace(T.scatter(len(t), x=t, y=q['microprice'], name='Microprice', mode='lines', line=dict(color=T.AQUA, width=1, shape='hv'), visible='legendonly'))
    tk = F.taker_orders(tape, window)
    if len(tk) > MAX_TRADE_POINTS:
        tk = tk.nlargest(MAX_TRADE_POINTS, 'qty').sort_values('t_s')
    size = 5 + 7 * np.sqrt(tk['qty'] / max(tk['qty'].max(), 1))
    for sign, name, color, symbol in ((1, 'Aggressive buy', T.BUY, 'triangle-up'), (-1, 'Aggressive sell', T.SELL, 'triangle-down')):
        m = tk['sign'] == sign
        price.add_trace(
            T.scatter(
                int(m.sum()),
                x=tk.loc[m, 't_s'],
                y=tk.loc[m, 'worst_price'],
                name=name,
                mode='markers',
                marker=dict(color=color, size=size[m], symbol=symbol, line=dict(color=T.SURFACE, width=1)),
                customdata=np.stack([tk.loc[m, 'qty'], tk.loc[m, 'depth']], axis=-1),
                hovertemplate='%{y}  qty %{customdata[0]}  levels past touch %{customdata[1]}<extra></extra>',
            )
        )

    spread = T.figure('Spread over time', x='time (s)', y='ticks', hovermode='x unified', showlegend=False)
    spread.add_trace(T.scatter(len(t), x=t, y=q['spread'], name='Spread', mode='lines', line=dict(color=T.ORANGE, width=1.5, shape='hv')))

    hist = F.time_weighted_hist(tape, 'spread', window)
    shist = T.figure('Spread distribution (share of time)', x='spread (ticks)', y='share of time', showlegend=False)
    shist.add_trace(go.Bar(x=hist.index, y=hist.values, marker=dict(color=T.ORANGE), hovertemplate='%{x} ticks: %{y:.1%}<extra></extra>'))
    shist.update_yaxes(tickformat='.0%')
    shist.update_xaxes(dtick=1)

    times, prices, z = F.depth_heatmap(tape, window)
    heat = T.figure('Resting depth by price (bids +, asks -)', x='time (s)', y='price (ticks)', height=420)
    if z.size:
        lim = max(float(np.abs(z).max()), 1.0)
        heat.add_trace(
            go.Heatmap(
                x=times, y=prices, z=z, zmin=-lim, zmax=lim, zmid=0, colorscale=T.DIVERGING,
                colorbar=dict(title=dict(text='qty', font=dict(color=T.MUTED)), tickfont=dict(color=T.MUTED)),
                hovertemplate='t %{x:.1f}s  price %{y}  qty %{z}<extra></extra>',
            )
        )

    note = 'Zoom the timeline above to see every quote change.' if sampled else 'Every quote change in the window is drawn.'
    return T.grid(
        T.panel(T.graph(price), f'{note} Marker size grows with order quantity; it sits at the worst price the order reached.', wide=True),
        T.panel(T.graph(spread)),
        T.panel(T.graph(shist), 'A market maker quoting inside the spread needs it wider than 1 tick most of the time.'),
        T.panel(T.graph(heat), 'Snapshots every 100 ms. Look for walls (persistent large levels) and how deep the book is.', wide=True),
    )
