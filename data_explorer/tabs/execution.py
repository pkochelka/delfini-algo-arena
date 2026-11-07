"""Queue outcomes, conditional maker markouts, and hypothetical cancellation risk."""
import numpy as np
import plotly.graph_objects as go
from dash import html

from analysis import execution as E
from data_explorer import theme as T


def _pct(value):
    return f'{100 * value:.2f}%' if np.isfinite(value) else '-'


def _table(rows, columns):
    return html.Div(html.Table([
        html.Thead(html.Tr([html.Th(label) for _, label, _ in columns])),
        html.Tbody([html.Tr([html.Td(fmt(row.get(key))) for key, _, fmt in columns]) for row in rows]),
    ], className='stats'), style={'overflowX': 'auto', 'padding': '8px'})


def _bars(fig, table, columns):
    for (column, label), color in zip(columns, T.SERIES):
        fig.add_trace(go.Bar(x=table['queue'], y=table[column], name=label, marker_color=color,
                            customdata=table.orders,
                            hovertemplate='%{x} lots ahead<br>%{y:.1%}<br>Orders: %{customdata:,}<extra>%{fullData.name}</extra>'))
    fig.update_layout(barmode='group')
    fig.update_yaxes(tickformat='.0%', range=[0, 1])


def render_queue(tape, window, dt, *, side='both', size='all', placement='touch', horizon=1.0, **_):
    stats, summary = E.queue_fills(tape, window, horizon, side, size, placement)
    fig = T.figure(f'Historical fill probability within {horizon:g} s',
                   x='Lots ahead at the same price on entry', y='Share of orders')
    fraction = T.figure('Average fraction of original quantity filled',
                        x='Lots ahead at the same price on entry', y='Order-weighted fraction')
    if stats.empty:
        T.empty(fig, 'No eligible passive orders with complete follow-up. Widen the window or change the filters.')
        T.empty(fraction, 'No eligible orders')
    else:
        _bars(fig, stats, [('any_fill', 'Any fill'), ('full_fill', 'Full fill')])
        _bars(fraction, stats, [('mean_fraction', 'Mean fraction filled')])
    columns = [('queue', 'Lots ahead', str), ('orders', 'Orders', T.fmt),
               ('any_fill', 'Any fill', _pct), ('full_fill', 'Full fill', _pct),
               ('mean_fraction', 'Mean fraction', _pct),
               ('median_first_ms', 'Median first fill (ms)*', T.fmt)]
    return html.Div([
        html.P('Observed FIFO outcomes of historical passive orders, not a forecast for a newly submitted quote. '
               'No additional 5 ms arrival delay is applied to these already-accepted orders.', className='note'),
        html.Div([T.tile('Eligible orders', summary['orders']), T.tile('Any fill', _pct(summary['any_fill'])),
                  T.tile('Full fill', _pct(summary['full_fill'])),
                  T.tile('Tail orders excluded', summary['excluded_tail'], 'Need the full selected horizon')], className='tiles'),
        T.grid(T.panel(T.graph(fig), 'Queue ahead includes remaining volume at the same price. Cancellations ahead and FIFO executions are respected by replay.'),
               T.panel(T.graph(fraction), 'Each order has equal weight. Canceled orders remain in the denominator; their fills before cancellation still count.'),
               T.panel(_table(stats.to_dict('records'), columns),
                       '*Median time is conditional on receiving a fill. This is not an unconditional waiting-time estimate. '
                       'Size, placement, and historical cancellation choices can confound comparisons. Small groups may be noisy.', wide=True)),
    ])


def render_maker(tape, window, dt, *, side='both', size='all', horizon=1.0, **_):
    tables, summary = E.maker_markouts(tape, window, horizon, side, size)
    titles = {'spread': 'Pre-fill spread (ticks)', 'imbalance': 'Maker-side share of touch liquidity',
              'volatility': 'Trailing midpoint volatility regime', 'aggressor_size': 'Aggressor submitted size (lots)'}
    panels = []
    for field, table in tables.items():
        fig = T.figure(f'Maker markout by {titles[field].lower()}', x=titles[field], y='Ticks per filled lot')
        fig.add_hline(y=0, line=dict(color=T.MUTED, dash='dot'))
        if table.empty:
            T.empty(fig, 'No fills with a valid pre-fill book and complete follow-up in this window.')
        else:
            fig.add_trace(go.Bar(x=table.group, y=table['mean'], name='Volume-weighted mean', marker_color=T.BLUE,
                                customdata=table[['fills', 'takers', 'lots']].to_numpy(),
                                hovertemplate='%{x}<br>Mean: %{y:.3f} ticks<br>Fills: %{customdata[0]:,}'
                                              '<br>Aggressors: %{customdata[1]:,}<br>Lots: %{customdata[2]:,}<extra></extra>'))
            fig.add_trace(go.Scatter(x=table.group, y=table.p10, name='10th percentile',
                                    mode='markers', marker=dict(color=T.ORANGE, symbol='triangle-down', size=9)))
            fig.add_trace(go.Scatter(x=table.group, y=table.p90, name='90th percentile',
                                    mode='markers', marker=dict(color=T.AQUA, symbol='triangle-up', size=9)))
        panels.append(T.panel(T.graph(fig), 'Percentiles describe the volume-weighted outcome distribution, not confidence intervals. Positive means favorable for the passive maker.'))
    rows = [dict(condition=f'{titles[field]}: {row["group"]}', **row)
            for field, table in tables.items() for row in table.to_dict('records')]
    columns = [('condition', 'Condition', str), ('fills', 'Fills', T.fmt), ('takers', 'Aggressors', T.fmt),
               ('lots', 'Lots', T.fmt), ('mean', 'Mean ticks', T.fmt), ('p10', 'P10', T.fmt),
               ('median', 'Median', T.fmt), ('p90', 'P90', T.fmt), ('adverse', 'Negative lots', _pct)]
    return html.Div([
        html.P(f'Actual maker execution prices versus midpoint {horizon:g} s later: '
               '+(future mid - fill) for buys and -(future mid - fill) for sells. '
               'This is a markout diagnostic, not realized strategy PnL.', className='note'),
        html.Div([T.tile('Eligible fills', summary['fills']), T.tile('Filled lots', summary.get('lots', 0)),
                  T.tile('Mean maker markout', T.fmt(summary.get('mean')), 'Ticks per lot, volume weighted'),
                  T.tile('Negative-markout lots', _pct(summary.get('adverse', np.nan))),
                  T.tile('Tail fills excluded', summary['excluded_tail']),
                  T.tile('Invalid book excluded', summary.get('invalid_quotes', 0))], className='tiles'),
        T.grid(*panels, T.panel(_table(rows, columns),
               'Conditions use the book before the taker event, including correct ordering within equal timestamps. '
               'Maker-side imbalance is bid share for maker buys and ask share for maker sells. '
               'Volatility is a causal 30-second RMS of 1-second midpoint changes (at least 5 observations); '
               f'regime cutoffs in this window: {T.fmt(summary.get("vol_low"))} / {T.fmt(summary.get("vol_high"))} ticks/sqrt(s). '
               'Groups overlap across panels; do not sum them. Sparse groups are descriptive, not reliable signals.', wide=True)),
    ])


def render_latency(tape, window, dt, *, side='both', size='all', placement='touch', latency=5.0, cancel_age=100.0, **_):
    quotes, cancels, summary = E.latency_risk(tape, window, latency, cancel_age, side, size, placement, sample_s=dt)
    adverse = T.figure('Adverse midpoint move during the delay', x='Delay (ms)', y='Share of sampled decisions')
    cross = T.figure('Would the old touch quote now cross?', x='Delay (ms)', y='Share of sampled decisions')
    race = T.figure('Fill before a hypothetical cancel takes effect', x='Cancel latency (ms)', y='Share of live orders')
    for direction, color, name in [('B', T.BUY, 'Maker buy'), ('A', T.SELL, 'Maker sell')]:
        q = quotes[quotes.side == direction]
        c = cancels[cancels.side == direction]
        for fig, frame, field, count in [(adverse, q, 'adverse', 'samples'), (cross, q, 'marketable', 'samples'),
                                         (race, c, 'any_fill', 'orders')]:
            if len(frame) and frame[count].sum():
                fig.add_trace(go.Scatter(x=frame.delay_ms, y=frame[field], mode='lines+markers', name=name,
                                        line=dict(color=color), customdata=frame[count],
                                        hovertemplate='%{x:g} ms<br>%{y:.2%}<br>Observations: %{customdata:,}<extra>%{fullData.name}</extra>'))
    for fig in (adverse, cross, race):
        fig.update_yaxes(tickformat='.0%', range=[0, 1])
        fig.add_vline(x=latency, line=dict(color=T.MUTED, dash='dash'))
        if not fig.data:
            T.empty(fig, 'No eligible observations. Try a wider window or different order filters.')
    selected = cancels[np.isclose(cancels.delay_ms, latency)].copy()
    selected['side'] = selected.side.map({'B': 'Buy', 'A': 'Sell'})
    columns = [('side', 'Maker side', str), ('orders', 'Live orders', T.fmt),
               ('any_fill', 'Any fill while pending', _pct), ('full_fill', 'All remaining quantity filled', _pct),
               ('mean_lost_lots', 'Mean lots filled', T.fmt), ('lost_lots', 'Total lots filled', T.fmt)]
    return html.Div([
        html.P('Price charts measure observed changes over a delay. The cancellation chart is a historical '
               'what-if, not a measurement of our own cancellation failures.', className='note'),
        html.Div([T.tile('Selected latency', f'{latency:g} ms'), T.tile('Request cancel at age', f'{cancel_age:g} ms'),
                  T.tile('Live order cohort', summary['cancel_orders']),
                  T.tile('Price sampling', f'{summary["sample_s"]:g} s', 'At most 50,000 decision times')], className='tiles'),
        T.grid(T.panel(T.graph(adverse), 'A falling midpoint is adverse for a buy quote; a rising midpoint is adverse for a sell quote. This is not a fill probability.'),
               T.panel(T.graph(cross), 'Buy at the old best bid >= later ask, or sell at the old best ask <= later bid. Marketability is not guaranteed execution.'),
               T.panel(T.graph(race), 'For orders actually alive at the selected age, count their real subsequent maker fills before cancel activation. '
                       'A fill exactly at activation is excluded, since the simulator processes a ready cancel before that market event.', wide=True),
               T.panel(_table(selected.to_dict('records'), columns),
                       'Counts use the same complete-follow-up cohort across delays. Existing historical cancellations remain in force. '
                       'Order size and placement filters apply to cancel experiments; price charts use a time grid and do not model size. '
                       'These are separate what-if requests for each order, not a simultaneous strategy simulation.', wide=True)),
    ])
