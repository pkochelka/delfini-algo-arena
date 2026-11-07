"""CLI entry point: replay one or more market data CSVs and serve the data explorer.

    uv run data-explorer --data data/round1.csv [data/round2.csv ...] [--port 8051]
    uv run data-explorer --data data/round1.csv --export [--window T0 T1] [--dt 1]   # write data_explorer/round1/, no server

Zoom the timeline at the top to pick the analysis window. Every tab recomputes
for it, and double-clicking the timeline resets it to the whole tape.
"""

from __future__ import annotations

import argparse
import time
from functools import lru_cache
from pathlib import Path

import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html, no_update

from analysis import features as F
from analysis import load_tape
from analysis.replay import Tape
from data_explorer import theme as T
from data_explorer.tabs import compare, flow, liquidity, overview, predict, price, volatility

TABS = {
    'overview': ('Overview', overview.render),
    'price': ('Price & spread', price.render),
    'vol': ('Volatility', volatility.render),
    'liq': ('Liquidity & book', liquidity.render),
    'flow': ('Trades & flow', flow.render),
    'predict': ('Predictability', predict.render),
    'compare': ('Compare tapes', None),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Market data explorer: statistics of raw round CSVs')
    parser.add_argument('--data', type=Path, nargs='+', required=True, help='one or more tape CSVs (data/*.csv)')
    parser.add_argument('--port', type=int, default=8051)
    parser.add_argument('--refresh', action='store_true', help='rebuild the replay cache')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--export', action='store_true', help='write every tab to data_explorer/<tape>/ (README.md, csv/, report.html) and exit')
    parser.add_argument('--window', type=float, nargs=2, metavar=('T0', 'T1'), help='with --export: window in seconds (default whole tape)')
    parser.add_argument('--dt', type=float, default=1.0, help='with --export: the sampling selector, s (default 1)')
    parser.add_argument('--png', action='store_true', help='with --export: also write PNGs (needs kaleido and Chrome)')
    parser.add_argument('--all-csv', action='store_true', help='with --export: also write CSVs of the large charts (price paths, heatmaps)')
    return parser.parse_args()


def timeline_figure(tape: Tape) -> go.Figure:
    mid = F.mid_on_grid(tape, max(tape.duration_s / 3000, 0.1))
    fig = T.figure(height=150, showlegend=False, hovermode='x unified', dragmode='zoom')
    fig.update_layout(margin=dict(l=56, r=16, t=8, b=28), uirevision=tape.name)
    fig.add_trace(go.Scatter(x=mid.index, y=mid.values, name='Mid', mode='lines', line=dict(color=T.NEUTRAL, width=1.5), hovertemplate='%{x:.1f}s  mid %{y}<extra></extra>'))
    fig.update_yaxes(fixedrange=True, title_text=None)
    return fig


def build_app(tapes: dict[str, Tape]) -> Dash:
    app = Dash(__name__, title='Data explorer', suppress_callback_exceptions=True)
    names = list(tapes)

    @lru_cache(maxsize=64)
    def render_tab(tab: str, name: str, window: tuple[float, float] | None, dt: float):
        started = time.perf_counter()
        if tab == 'compare':
            out = compare.render(list(tapes.values()))
        else:
            out = TABS[tab][1](tapes[name], window, dt)
        print(f'  rendered {tab} for {name} {window or "whole tape"} dt={dt} in {time.perf_counter() - started:.2f}s', flush=True)
        return out

    app.layout = html.Div(
        [
            html.Div(
                [
                    html.H1('Market data explorer'),
                    html.Div(['Tape', dcc.Dropdown(names, names[0], id='tape', clearable=False)], className='control'),
                    html.Div(['Sampling', dcc.Dropdown([{'label': f'{d:g} s', 'value': d} for d in T.DT_OPTIONS], 1.0, id='dt', clearable=False)], className='control'),
                    html.Span(id='window-label', className='window-label'),
                    html.Button('Whole tape', id='reset', className='reset'),
                ],
                className='topbar',
            ),
            html.Div(dcc.Graph(id='timeline', figure=timeline_figure(tapes[names[0]]), config={'displaylogo': False, 'displayModeBar': False}), className='timeline'),
            html.P('Drag across the timeline to analyse a window; double-click it or press "Whole tape" to reset.', className='timeline-hint'),
            dcc.Store(id='window', data=None),
            dcc.Location(id='url', refresh=False),
            dcc.Tabs(
                id='tab',
                value='overview',
                children=[dcc.Tab(label=label, value=key, className='tab', selected_className='tab--selected') for key, (label, _) in TABS.items()],
                className='tabs-row',
            ),
            dcc.Loading(html.Div(id='content'), type='dot', color=T.BLUE),
        ],
        className='app',
    )

    @app.callback(Output('tab', 'value'), Output('url', 'search'), Input('url', 'search'), Input('tab', 'value'))
    def _tab_url(search, tab):
        """Keep ?tab=<key> in the address bar in sync with the tab, so a tab can be linked or reloaded."""
        from urllib.parse import parse_qs

        from dash import ctx

        if ctx.triggered_id == 'url':
            wanted = parse_qs((search or '').lstrip('?')).get('tab', [None])[0]
            return (wanted if wanted in TABS else no_update), no_update
        return no_update, f'?tab={tab}'

    @app.callback(Output('timeline', 'figure'), Input('tape', 'value'))
    def _timeline(name: str):
        return timeline_figure(tapes[name])

    @app.callback(
        Output('window', 'data'),
        Input('timeline', 'relayoutData'),
        Input('reset', 'n_clicks'),
        Input('tape', 'value'),
        State('window', 'data'),
        prevent_initial_call=True,
    )
    def _window(relayout, _clicks, _name, current):
        from dash import ctx

        if ctx.triggered_id in ('reset', 'tape'):
            return None
        if not relayout:
            return no_update
        if relayout.get('xaxis.autorange'):
            return None
        if 'xaxis.range[0]' in relayout:
            return [round(float(relayout['xaxis.range[0]']), 1), round(float(relayout['xaxis.range[1]']), 1)]
        if 'xaxis.range' in relayout:
            return [round(float(v), 1) for v in relayout['xaxis.range']]
        return current

    @app.callback(Output('timeline', 'figure', allow_duplicate=True), Input('reset', 'n_clicks'), State('tape', 'value'), prevent_initial_call=True)
    def _reset_timeline(_clicks, name):
        return timeline_figure(tapes[name])

    @app.callback(Output('window-label', 'children'), Input('window', 'data'), Input('tape', 'value'))
    def _label(window, name):
        t0, t1 = F.bounds(tapes[name], window)
        return f'Window {t0:,.1f}–{t1:,.1f} s ({(t1 - t0) / 60:.1f} min)'

    @app.callback(Output('content', 'children'), Input('tab', 'value'), Input('tape', 'value'), Input('window', 'data'), Input('dt', 'value'))
    def _content(tab, name, window, dt):
        return render_tab(tab, name, tuple(window) if window else None, float(dt))

    return app


def main() -> None:
    args = parse_args()
    if args.export:
        from data_explorer.export import run

        run(args.data, tuple(args.window) if args.window else None, args.dt, refresh=args.refresh, png=args.png, all_csv=args.all_csv)
        return
    tapes: dict[str, Tape] = {}
    for path in args.data:
        tape = load_tape(path, refresh=args.refresh)
        tapes[tape.name] = tape
    app = build_app(tapes)
    print(f'Serving on http://127.0.0.1:{args.port}', flush=True)
    app.run(debug=args.debug, port=args.port, dev_tools_ui=args.debug)


if __name__ == '__main__':
    main()
