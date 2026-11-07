"""Liquidity & book shape: depth profile, touch sizes, imbalance, order sizes, lifetimes, placement."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from dash import html

from analysis import features as F
from analysis.replay import Tape
from data_explorer import theme as T


def render(tape: Tape, window: F.Window, dt: float) -> html.Div:
    dp = F.depth_profile(tape, window)
    depth = T.figure('Mean resting quantity by distance from the touch', x='ticks behind the best price', y='mean qty', barmode='group')
    depth.add_trace(go.Bar(x=dp['ticks'], y=dp['bid'], name='Bid', marker=dict(color=T.BID)))
    depth.add_trace(go.Bar(x=dp['ticks'], y=dp['ask'], name='Ask', marker=dict(color=T.ASK)))

    touch = T.figure('Quantity at the best price (share of time)', x='qty', y='share of time', barmode='overlay')
    for col, name, color in (('bid_qty', 'Bid', T.BID), ('ask_qty', 'Ask', T.ASK)):
        h = F.time_weighted_hist(tape, col, window)
        touch.add_trace(go.Bar(x=h.index, y=h.values, name=name, marker=dict(color=color), opacity=0.7))
    touch.update_yaxes(tickformat='.0%')

    t = F.grid(tape, max(dt, 0.1), window)
    imb = F.asof(tape, t, 'imbalance')
    imbf = T.figure('Queue imbalance  Qbid / (Qbid + Qask)', x='imbalance', y='share of samples', showlegend=False)
    imb = imb[~np.isnan(imb)]
    if len(imb):
        counts, edges = np.histogram(imb, np.linspace(0, 1, 21))
        imbf.add_trace(go.Bar(x=(edges[:-1] + edges[1:]) / 2, y=counts / len(imb), marker=dict(color=T.AQUA)))
        imbf.update_yaxes(tickformat='.0%')

    o = F.orders_in(tape, window)
    sizes = T.figure('Order sizes at insert', x='quantity', y='count', barmode='overlay')
    if len(o):
        hi = max(int(np.percentile(o['volume'], 99.5)), 1)
        bins = dict(start=0.5, end=hi + 0.5, size=max(1, hi // 60))
        for kind, color in (('passive', T.BLUE), ('aggressive', T.ORANGE)):
            v = o.loc[o['kind'] == kind, 'volume']
            sizes.add_trace(go.Histogram(x=v, name=f'{kind} ({len(v):,})', marker=dict(color=color), opacity=0.75, xbins=bins))

    lt = F.order_lifetimes(tape, window)
    life = T.figure('Lifetime of resting orders', x='seconds (log)', y='count')
    sub = ''
    if len(lt):
        lv = lt['lifetime_s'].to_numpy()
        pos = lv[lv > 0]
        if len(pos):
            edges = np.logspace(np.log10(pos.min()), np.log10(pos.max() + 1e-9), 40)
            for end, color in (('cancelled', T.BLUE), ('filled', T.AQUA)):
                v = lt.loc[lt['end'] == end, 'lifetime_s'].to_numpy()
                T.log_hist(life, v, end, color, edges)
            life.update_xaxes(type='log')
        sub = f'median {np.median(lv):.3g}s; {np.mean(lv < 0.01):.0%} live under 10 ms (fleeting quotes).'

    passive = o[~o['crossed']]
    place = T.figure('Where passive orders are placed', x='ticks from the same-side best (+ inside the spread, − behind)', y='count', showlegend=False)
    d = passive['dist'].dropna()
    if len(d):
        lo, hi = np.percentile(d, [0.5, 99.5])
        vc = d[(d >= lo) & (d <= hi)].value_counts().sort_index()
        place.add_trace(go.Bar(x=vc.index, y=vc.values, marker=dict(color=T.BLUE)))
    else:
        T.empty(place, 'Every passive order arrived with its own side of the book empty')

    return T.grid(
        T.panel(T.graph(depth), 'How much volume sits behind the touch: thin books mean large orders sweep several levels.', wide=True),
        T.panel(T.graph(touch), 'Small, frequently refreshed touch sizes mean our queue position matters less; large ones mean we wait behind them.'),
        T.panel(T.graph(imbf), 'Mass near 0 and 1 means one side is often nearly empty: the strongest imbalance signals.'),
        T.panel(T.graph(sizes), 'Compare with our ORDER_SIZE: quoting much larger than the crowd sizes is unusual for this tape.'),
        T.panel(T.graph(life), sub),
        T.panel(T.graph(place), '0 joins the best price, +1 improves it by a tick, −1 sits one tick behind.'),
    )
