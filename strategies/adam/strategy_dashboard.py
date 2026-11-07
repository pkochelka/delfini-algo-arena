"""Local comparison dashboard. Run with --prepare once, then --runs <folder>."""
import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPO = next(p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').exists() and (p / 'simulator').is_dir())
sys.path.insert(0, str(REPO))

import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, dash_table, dcc, html
from explorer.data.orderbook import build_best_quotes

INFO = {
    'basic_touch': ('Touch', '#2563eb', 'Fixed-size bids and asks at the market touch.',
                    'Tune BASE_SIZE first, then REFRESH_US and REQUOTE_TICKS. Larger size increases exposure; faster repricing sacrifices queue priority.'),
    'basic_inventory': ('Inventory', '#08916a', 'Moves quote prices and new-order sizes to reduce accumulated inventory.',
                        'Tune INVENTORY_SKEW_TICKS and SIZE_SKEW separately. Stronger skew reduces exposure but can sacrifice spread income.'),
    'basic_imbalance': ('Imbalance', '#8b5cf6', 'Adds a directional shift based on bid versus ask volume at the touch.',
                        'Tune IMBALANCE_TICKS. Set it to zero to compare with the inventory strategy. Validate the signal on later data after accounting for latency.'),
    'basic_volatility': ('Volatility', '#e38a16', 'Widens inventory-aware quotes when estimated midpoint volatility rises.',
                         'Tune VOL_MULT and VOL_TAU_S. If almost no trades occur, reduce VOL_MULT. Shorter memory reacts faster but is noisier.'),
}


def prepare(data):
    target = HERE / 'strategy_runs' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    target.mkdir(parents=True)
    for name in INFO:
        run = target / name
        run.mkdir()
        command = [sys.executable, '-B', '-m', 'simulator.main', '--strategy',
                   str(REPO / 'strategies' / 'adam' / 'models' / f'{name}.py'), '--backtest', str(data),
                   '--log-dir', str(run)]
        with (run / 'console.log').open('w', encoding='utf-8') as output:
            subprocess.run(command, cwd=REPO, stdout=output, stderr=subprocess.STDOUT, check=True)
        print(f'Prepared {name}', flush=True)
    (HERE / 'latest_strategy_runs.txt').write_text(str(target), encoding='utf-8')
    return target


def chart(title, y_title):
    fig = go.Figure()
    fig.update_layout(template='plotly_white', title=title, height=360,
                      margin=dict(l=55, r=20, t=55, b=45),
                      paper_bgcolor='white', plot_bgcolor='white',
                      font=dict(family='Segoe UI, Arial', color='#22304a'),
                      legend=dict(orientation='h', y=1.12), hovermode='x unified')
    fig.update_xaxes(title_text='Elapsed seconds', gridcolor='#edf1f7')
    fig.update_yaxes(title_text=y_title, gridcolor='#edf1f7')
    return fig


def load_run(folder, name):
    run = folder / name
    result = json.loads((run / 'result.json').read_text())
    metadata = json.loads((run / 'metadata.json').read_text())
    pnl = pd.read_csv(run / 'pnl.csv')
    trades = pd.read_csv(run / 'trades.csv')
    deltas = pd.read_csv(run / 'book_deltas.csv')
    # Stable ordering preserves fill order when timestamps coincide.
    trades = trades.sort_values('timestamp_us', kind='stable')
    start = int(deltas['timestamp_us'].min())
    end = int(max(deltas['timestamp_us'].max(), trades['timestamp_us'].max())) if len(trades) else int(deltas['timestamp_us'].max())
    own = trades[(trades.taker_owner == 'user') | (trades.maker_owner == 'user')].copy()
    own['side'] = [row.taker_side if row.taker_owner == 'user' else
                   ('A' if row.taker_side == 'B' else 'B') for row in own.itertuples()]
    own['signed_qty'] = own['quantity'].where(own['side'] == 'B', -own['quantity'])
    own['inventory'] = own['signed_qty'].cumsum()
    own['elapsed_s'] = (own['timestamp_us'] - start) / 1_000_000
    max_inventory = int(own['inventory'].abs().max()) if len(own) else 0
    assert max_inventory <= 80 and result['breach_penalty'] == 0
    assert int(own['signed_qty'].sum()) == result['inventory']
    quotes = build_best_quotes(deltas[deltas.owner == 'market'])
    quotes['elapsed_s'] = (quotes['timestamp_us'] - start) / 1_000_000
    return dict(result=result, metadata=metadata, pnl=pnl, own=own, quotes=quotes,
                duration=(end-start)/1_000_000, max_inventory=max_inventory)


def build_app(folder):
    runs = {name: load_run(folder, name) for name in INFO}
    app = Dash(__name__, title='Strategy comparison', update_title=None)
    app.index_string = '''<!DOCTYPE html><html><head>{%metas%}<title>{%title%}</title>
    {%favicon%}{%css%}<style>
    body{margin:0;background:#f3f6fa;color:#22304a;font-family:Segoe UI,Arial,sans-serif}
    .page{max-width:1440px;margin:auto;padding:30px}.eyebrow{text-transform:uppercase;letter-spacing:2px;color:#64748b;font-size:12px}
    h1{font-size:32px;margin:8px 0}h2{font-size:19px;margin:0 0 15px}.subtitle{color:#64748b;line-height:1.6}
    .warning{background:#fff8e8;border:1px solid #f0daa5;border-radius:10px;padding:14px 18px;margin:20px 0;line-height:1.6}
    .panel{background:white;border:1px solid #e1e7f0;border-radius:14px;padding:20px;margin:18px 0;box-shadow:0 3px 10px #14213d04}
    .grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.grid .panel{margin:0}.controls{display:flex;align-items:center;gap:20px;flex-wrap:wrap}
    .metrics{display:flex;gap:28px;flex-wrap:wrap;margin-top:18px}.metric strong{font-size:26px;display:block}.metric span{font-size:12px;color:#64748b}
    .notes{line-height:1.7;color:#536279}code{background:#edf2f8;padding:2px 5px;border-radius:4px}
    @media(max-width:850px){.grid{grid-template-columns:1fr}.page{padding:15px}h1{font-size:26px}}
    </style></head><body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer></body></html>'''
    summary = []
    for name, run in runs.items():
        r = run['result']
        summary.append({'Strategy': INFO[name][0], 'PnL': round(r['pnl'], 2),
                        'Volume': r['traded_volume'], 'Filled orders': r['fills'],
                        'Max |inventory|': run['max_inventory'], 'Final inventory': r['inventory'],
                        'Penalty': r['breach_penalty']})
    table_style = dict(style_table={'overflowX': 'auto'},
                       style_cell={'fontFamily': 'Segoe UI', 'padding': '12px', 'textAlign': 'right',
                                   'border': 'none', 'borderBottom': '1px solid #edf1f7'},
                       style_header={'fontWeight': 600, 'backgroundColor': '#f5f7fb'},
                       style_cell_conditional=[{'if': {'column_id': 'Strategy'}, 'textAlign': 'left'}])
    overlay = chart('PnL through time — all four strategies', 'PnL (simulator price units × lots)')
    overlay.update_layout(height=410)
    for name, run in runs.items():
        label, color = INFO[name][:2]
        overlay.add_trace(go.Scatter(x=run['pnl'].elapsed_s, y=run['pnl'].pnl,
                                    name=label, mode='lines', line=dict(color=color, shape='hv')))
    app.layout = html.Div([
        html.Div('Qminers / local research', className='eyebrow'),
        html.H1('Four strategies. One tape.'),
        html.Div('Compare results, inspect fills, and understand the risk each strategy took.', className='subtitle'),
        html.Div([
            html.Strong('Synthetic sample only. '),
            'These are baseline smoke tests, not a ranking for real competition data. ',
            html.Span('Click a legend entry to hide a curve; double-click to isolate it.'),
        ], className='warning'),
        html.Div([html.H2('Results at a glance'), dash_table.DataTable(
            id='summary', data=summary, columns=[{'name': k, 'id': k} for k in summary[0]],
            sort_action='native', style_data_conditional=[
                {'if': {'filter_query': '{PnL} > 0', 'column_id': 'PnL'}, 'color': '#087f5b', 'fontWeight': 600},
                {'if': {'filter_query': '{PnL} < 0', 'column_id': 'PnL'}, 'color': '#c43a3a', 'fontWeight': 600},
            ], **table_style)], className='panel'),
        html.Div(dcc.Graph(id='pnl-overlay', figure=overlay), className='panel'),
        html.Div([
            html.Div([html.Strong('Inspect a strategy'), dcc.Dropdown(
                id='strategy', options=[{'label': INFO[k][0], 'value': k} for k in INFO],
                value='basic_inventory', clearable=False, style={'width': '260px'})], className='controls'),
            html.Div(id='metrics', className='metrics'),
            html.Div(id='strategy-notes', className='notes'),
        ], className='panel'),
        html.Div([
            html.Div(dcc.Graph(id='inventory'), className='panel'),
            html.Div(dcc.Graph(id='fills-chart'), className='panel'),
        ], className='grid'),
        html.Div([html.H2('Individual fills'), dash_table.DataTable(
            id='fills-table', columns=[{'name': k, 'id': k} for k in
                                      ['Time (s)', 'Side', 'Price', 'Quantity', 'Inventory after fill']],
            page_size=10, sort_action='native', **table_style)], className='panel'),
        html.Div([
            html.P('Strategy files: strategies_a/basic_*.py. Internal exposure cap: ±80; competition cap: ±100. A pending cancel continues to reserve capacity.'),
            html.P('Full logs: ' + str(folder)),
            html.P('Detailed explanations and tuning grids: strategies/adam/basic_strategies_guide.md. No strategies have been submitted to the competition.'),
        ], className='notes'),
    ], className='page')

    @app.callback(Output('metrics', 'children'), Output('strategy-notes', 'children'),
                  Output('inventory', 'figure'), Output('fills-chart', 'figure'),
                  Output('fills-table', 'data'), Input('strategy', 'value'))
    def select(name):
        run = runs[name]
        r = run['result']
        label, color, description, tuning = INFO[name]
        own = run['own']
        metrics = [html.Div([html.Strong(str(value)), html.Span(title)], className='metric')
                   for title, value in [('Final PnL', round(r['pnl'], 2)), ('Traded lots', r['traded_volume']),
                                        ('Maximum |inventory|', run['max_inventory']), ('Breach penalty', r['breach_penalty'])]]
        inventory = chart(f'{label}: inventory', 'Lots (positive = long)')
        inventory.add_trace(go.Scatter(x=[0, *own.elapsed_s.tolist(), run['duration']],
                                      y=[0, *own.inventory.tolist(), r['inventory']], mode='lines',
                                      name='Inventory', line=dict(color=color, shape='hv')))
        for level in (-100, 100):
            inventory.add_hline(y=level, line_color='#d83e46', line_dash='dash')
        for level in (-80, 80):
            inventory.add_hline(y=level, line_color='#b5bfcc', line_dash='dot')
        inventory.update_yaxes(range=[-110, 110])
        prices = chart(f'{label}: market quotes and our fills', 'Price (ticks)')
        # Limit rendering load on longer tapes; fill markers are never downsampled.
        q = run['quotes']
        stride = max(1, len(q)//8000)
        q = pd.concat([q.iloc[::stride], q.tail(1)]).drop_duplicates('timestamp_us')
        for column, title, line_color in [('best_bid', 'Market bid', '#a3b4cc'), ('best_ask', 'Market ask', '#d0d9e5')]:
            prices.add_trace(go.Scatter(x=q.elapsed_s, y=q[column], mode='lines', name=title,
                                       line=dict(color=line_color, width=1)))
        for side, title, symbol, fill_color in [('B', 'Our buys', 'triangle-up', '#08916a'),
                                                ('A', 'Our sells', 'triangle-down', '#c43a3a')]:
            fills = own[own.side == side]
            prices.add_trace(go.Scatter(x=fills.elapsed_s, y=fills.price, name=title, mode='markers',
                                       customdata=fills['quantity'], marker=dict(symbol=symbol, size=9, color=fill_color),
                                       hovertemplate='%{x:.3f}s<br>Price %{y}<br>Quantity %{customdata}<extra>%{fullData.name}</extra>'))
        rows = [{'Time (s)': round(row.elapsed_s, 6), 'Side': 'Buy' if row.side == 'B' else 'Sell',
                 'Price': int(row.price), 'Quantity': int(row.quantity),
                 'Inventory after fill': int(row.inventory)} for row in own.itertuples()]
        notes = [html.P(description), html.P(tuning),
                 html.P('Gray dotted inventory lines are the internal cap; red dashed lines are the competition limit. Market quote lines are downsampled on large tapes; fill markers and inventory use every fill.')]
        return metrics, notes, inventory, prices, rows

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true', help='Generate new full-log sample backtests')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--data', type=Path, default=REPO / 'data' / 'sample.csv')
    parser.add_argument('--runs', type=Path)
    parser.add_argument('--port', type=int, default=8060)
    args = parser.parse_args()
    folder = prepare(args.data.resolve()) if args.prepare else args.runs
    if folder is None:
        folder = Path((HERE / 'latest_strategy_runs.txt').read_text(encoding='utf-8'))
    if args.prepare_only:
        print(folder, flush=True)
        return
    app = build_app(folder)
    print(f'Open http://127.0.0.1:{args.port}', flush=True)
    app.run(host='127.0.0.1', port=args.port, debug=False)


if __name__ == '__main__':
    main()
