"""Self-contained interactive HTML report over one or more simulator runs.

    uv run results-explorer logs/r1_ladder_mm                       # -> logs/r1_ladder_mm/report.html
    uv run results-explorer logs/a logs/b --open                    # comparison + a tab per run
    uv run results-explorer --all --out logs/all-runs.html          # every run directory under logs/

Full-logs runs (no --no-logs) get everything: inventory, PnL decomposition, markouts,
order and quote statistics. --no-logs runs get the PnL curve and result.json figures only.
Each run directory also gets results_summary.json with every metric.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import time
import webbrowser
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.offline
from plotly.subplots import make_subplots

from data_explorer.theme import AQUA, AXIS, BLUE, GRID, INK, INK_2, MUTED, ORANGE, PAGE, SERIES, SURFACE, figure, fmt
from results_explorer.load import load_run
from results_explorer.metrics import MARKOUT_HORIZONS_S, Analysis, analyse

NEG = '#e66767'
YELLOW = '#c98500'

CSS = f"""
body {{ margin:0; background:{PAGE}; color:{INK}; font-family:system-ui,-apple-system,"Segoe UI",sans-serif; }}
.app {{ padding:16px; max-width:1600px; margin:0 auto; }}
h1 {{ font-size:18px; font-weight:600; margin:0 0 4px; }}
h2 {{ font-size:16px; font-weight:600; margin:4px 0 4px; }}
.meta {{ font-size:12px; color:{MUTED}; margin-bottom:10px; line-height:1.6; word-break:break-all; }}
.meta b {{ color:{INK_2}; font-weight:500; }}
.warn {{ font-size:12px; color:{YELLOW}; margin:4px 0; }}
.section-title {{ font-size:13px; font-weight:600; color:{INK_2}; text-transform:uppercase; letter-spacing:.04em; margin:18px 2px 8px; }}
.tiles {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(150px,1fr)); gap:8px; }}
.tile {{ background:{SURFACE}; border-radius:8px; padding:10px 12px; cursor:help; }}
.tile-label {{ font-size:11px; color:{MUTED}; text-transform:uppercase; letter-spacing:.04em; }}
.tile-value {{ font-size:20px; font-weight:600; margin-top:4px; font-variant-numeric:tabular-nums; }}
.tile-value.neg {{ color:{NEG}; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(min(520px,100%),1fr)); gap:12px; }}
.panel {{ background:{SURFACE}; border-radius:8px; padding:4px 4px 8px; min-width:0; }}
.panel.wide {{ grid-column:1/-1; }}
.note {{ font-size:12px; color:{MUTED}; margin:2px 12px 4px; line-height:1.45; }}
.table-wrap {{ overflow-x:auto; background:{SURFACE}; border-radius:8px; }}
table {{ border-collapse:collapse; width:100%; font-size:12px; font-variant-numeric:tabular-nums; }}
th, td {{ padding:6px 10px; text-align:right; border-bottom:1px solid {GRID}; white-space:nowrap; }}
th {{ color:{MUTED}; font-weight:500; position:sticky; top:0; background:{SURFACE}; }}
td:first-child, th:first-child {{ text-align:left; }}
td.neg {{ color:{NEG}; }} td.best {{ color:{AQUA}; font-weight:600; }}
.tabs {{ display:flex; flex-wrap:wrap; gap:6px; margin:12px 0 4px; position:sticky; top:0; background:{PAGE}; padding:6px 0; z-index:10; }}
.tab {{ background:{SURFACE}; color:{INK_2}; border:1px solid {AXIS}; border-radius:6px; padding:6px 12px; font-size:13px; cursor:pointer; }}
.tab.active {{ color:{INK}; border-color:{BLUE}; }}
.page {{ display:none; }} .page.active {{ display:block; }}
"""

JS = """
function showTab(id) {
  document.querySelectorAll('.page').forEach(p => p.classList.toggle('active', p.id === id));
  document.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t.dataset.page === id));
  document.querySelectorAll('#' + id + ' .js-plotly-plot').forEach(g => Plotly.Plots.resize(g));
  history.replaceState(null, '', '#' + id);
}
window.addEventListener('load', () => {
  const first = document.querySelector('.page');
  const want = location.hash.slice(1);
  showTab(want && document.getElementById(want) ? want : first.id);
});
"""

# (key, label, kind, help). kind: num | pct | ticks | ms | s | sign (num, red when negative)
TILES: dict[str, list[tuple[str, str, str, str]]] = {
    'Performance': [
        ('pnl', 'PnL', 'sign', "The simulator's final PnL (cash + liquidation markout − penalty)."),
        ('pnl_se', 'PnL s.e.', 'num', "The simulator's standard error of the final PnL."),
        ('pnl_per_hour', 'PnL / hour', 'sign', 'Final PnL divided by run length in hours.'),
        ('max_drawdown', 'Max drawdown', 'sign', 'Largest peak-to-trough fall of the 1 s PnL curve.'),
        ('pnl_to_max_dd', 'PnL / max DD', 'num', 'Final PnL over the max drawdown.'),
        ('minute_sharpe', 'Sharpe / min', 'num', 'Mean over std of per-minute PnL changes.'),
        ('minute_tstat', 't-stat', 'num', 'Per-minute Sharpe × √minutes: how many standard errors the total is from zero.'),
        ('minute_win_rate', 'Winning minutes', 'pct', 'Share of minutes with positive PnL change.'),
        ('worst_minute', 'Worst minute', 'sign', 'Most negative one-minute PnL change.'),
    ],
    'Inventory & risk': [
        ('mean_abs_inventory', 'Mean |inv|', 'num', 'Time-weighted mean absolute inventory.'),
        ('abs_inventory_q95', '|inv| q95', 'num', "Time-weighted 95th percentile of |inventory| (simulator)."),
        ('time_at_limit_frac', 'Time at limit', 'pct', 'Share of time with |inventory| ≥ position limit.'),
        ('time_flat_frac', 'Time flat', 'pct', 'Share of time with zero inventory.'),
        ('mean_inventory', 'Mean inv', 'num', 'Time-weighted mean signed inventory (directional bias).'),
        ('breach_seconds', 'Breach time (s)', 'num', 'Seconds with |inventory| above the limit.'),
        ('breach_penalty', 'Breach penalty', 'num', "The simulator's total breach penalty."),
        ('final_inventory', 'Final inv', 'num', 'Inventory at the end of the run.'),
    ],
    'Execution': [
        ('fills', 'Fills', 'num', 'Trades the user took part in (one per maker order matched).'),
        ('traded_volume', 'Volume', 'num', 'Total units bought and sold.'),
        ('maker_volume_share', 'Maker share', 'pct', 'Share of volume traded passively.'),
        ('pnl_per_unit_traded', 'PnL / unit', 'num', 'Final PnL per unit traded.'),
        ('edge_ticks_per_unit', 'Edge / unit', 'ticks', 'Volume-weighted side·(mid just before the fill − price): spread captured, in ticks.'),
        ('markout_1s_ticks', 'Markout 1 s', 'ticks', 'Volume-weighted side·(mid 1 s after − price), ticks per unit.'),
        ('markout_60s_ticks', 'Markout 60 s', 'ticks', 'Volume-weighted side·(mid 60 s after − price), ticks per unit.'),
        ('edge_total', 'Spread edge', 'sign', 'Σ edge × qty: what fills earned against the mid at the moment of trading.'),
        ('inventory_pnl_total', 'Inventory PnL', 'sign', 'Mark-to-mid PnL − spread edge: gains/losses from holding positions after fills.'),
        ('liquidation_total', 'Liquidation', 'sign', 'Sim PnL − mark-to-mid + penalty: cost of marking the final position to the book, not the mid.'),
    ],
    'Orders & quoting': [
        ('orders_sent', 'Orders sent', 'num', 'add_sent events.'),
        ('orders_rejected', 'Rejected', 'num', 'add_failed events (position-limit check at acceptance).'),
        ('orders_with_fill_frac', 'Orders filled', 'pct', 'Share of accepted orders with at least one fill.'),
        ('qty_fill_ratio', 'Qty fill ratio', 'pct', 'Filled quantity over accepted quantity.'),
        ('orders_per_fill', 'Orders / fill', 'num', 'Orders sent per fill.'),
        ('cancels_failed', 'Cancels lost', 'num', 'Cancels that arrived after the order was already gone (lost the latency race).'),
        ('median_order_lifetime_ms', 'Median life', 'ms', 'Median time from acceptance to fill or cancel.'),
        ('two_sided_frac', 'Two-sided', 'pct', 'Share of 1 s samples with a resting user bid and ask.'),
        ('bid_presence_frac', 'Bid present', 'pct', 'Share of samples with a resting user bid.'),
        ('ask_presence_frac', 'Ask present', 'pct', 'Share of samples with a resting user ask.'),
        ('bid_at_or_inside_touch_frac', 'Bid at touch', 'pct', 'When bidding: share of samples at or inside the market best bid.'),
    ],
}

COMPARE_ROWS = [k for group in TILES.values() for k, *_ in group]
HIGHER_BETTER = {'pnl', 'pnl_per_hour', 'max_drawdown', 'pnl_to_max_dd', 'minute_sharpe', 'minute_tstat', 'minute_win_rate',
                 'worst_minute', 'edge_ticks_per_unit', 'markout_1s_ticks', 'markout_60s_ticks', 'pnl_per_unit_traded'}
LOWER_BETTER = {'pnl_se', 'mean_abs_inventory', 'abs_inventory_q95', 'time_at_limit_frac', 'breach_seconds', 'breach_penalty'}


def _format(value, kind: str) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return '-'
    if kind == 'pct':
        return f'{100 * value:.1f}%'
    if kind == 'ticks':
        return f'{value:+.3f}'
    if kind == 'ms':
        return f'{value:,.0f} ms'
    if kind in ('num', 'sign') and isinstance(value, float) and abs(value) >= 100:
        return f'{value:,.0f}'
    return fmt(value)


def _is_neg(value, kind: str) -> bool:
    return kind in ('sign', 'ticks') and isinstance(value, (int, float)) and value < 0


def _div(fig: go.Figure) -> str:
    return plotly.offline.plot(fig, output_type='div', include_plotlyjs=False,
                               config={'displaylogo': False, 'responsive': True})


def _panel(fig: go.Figure, note: str | None = None, wide: bool = False) -> str:
    n = f'<div class="note">{note}</div>' if note else ''
    return f'<div class="panel{" wide" if wide else ""}">{_div(fig)}{n}</div>'


def _tiles(s: dict) -> str:
    out = []
    for group, specs in TILES.items():
        cells = []
        for key, label, kind, help_ in specs:
            if key not in s:
                continue
            v = s[key]
            cls = ' neg' if _is_neg(v, kind) else ''
            cells.append(f'<div class="tile" title="{html.escape(help_)}"><div class="tile-label">{label}</div>'
                         f'<div class="tile-value{cls}">{_format(v, kind)}</div></div>')
        if cells:
            out.append(f'<div class="section-title">{group}</div><div class="tiles">{"".join(cells)}</div>')
    return ''.join(out)


# ---------------------------------------------------------------- per-run figures

def _timeline(a: Analysis) -> go.Figure:
    g, run = a.grid, a.run
    full = run.full_logs and 'edge' in g
    has_price = 'mid' in g
    rows: list[tuple[str, float]] = [('PnL', 0.34)]
    if run.full_logs:
        rows.append(('Inventory', 0.2))
    if has_price:
        rows.append(('Price', 0.3))
    rows.append(('Drawdown', 0.16))
    fig = make_subplots(rows=len(rows), cols=1, shared_xaxes=True, vertical_spacing=0.025,
                        row_heights=[h for _, h in rows])
    base = figure(height=260 * len(rows) if len(rows) > 2 else 520)
    fig.update_layout(base.layout)
    fig.update_layout(hovermode='x unified', margin=dict(l=64, r=16, t=70, b=40), legend=dict(y=1.01))
    axis = dict(gridcolor=GRID, linecolor=AXIS, zerolinecolor=AXIS, tickfont=dict(color=MUTED), title_font=dict(color=MUTED))
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    row = {name: i + 1 for i, (name, _) in enumerate(rows)}
    x = g['t_s']

    def line(name, y, color, r, **kw):
        fig.add_trace(go.Scattergl(x=x, y=y, name=name, mode='lines', line=dict(color=color, width=kw.pop('width', 1.4), **kw.pop('line', {})), **kw), row=r, col=1)

    line('Sim PnL', g['pnl'], INK, row['PnL'])
    if full:
        line('Spread edge', g['edge'], AQUA, row['PnL'])
        line('Inventory PnL', g['inventory_pnl'], ORANGE, row['PnL'])
        line('Liquidation + penalty', g['liquidation'] + g['penalty'], NEG, row['PnL'], visible='legendonly')
    if run.full_logs:
        r = row['Inventory']
        line('Inventory', g['inventory'], BLUE, r, line=dict(shape='hv'), fill='tozeroy', fillcolor='rgba(57,135,229,0.15)')
        for lim in (run.position_limit, -run.position_limit):
            fig.add_hline(y=lim, line=dict(color=NEG, width=1, dash='dot'), row=r, col=1)
    if has_price:
        r = row['Price']
        line('Market bid', g['bid'], MUTED, r, width=1, line=dict(shape='hv'))
        line('Market ask', g['ask'], MUTED, r, width=1, line=dict(shape='hv', dash='dot'))
        if 'user_bid' in g:
            line('User bid', g['user_bid'], BLUE, r, width=1, line=dict(shape='hv'), opacity=0.8, visible='legendonly')
            line('User ask', g['user_ask'], ORANGE, r, width=1, line=dict(shape='hv'), opacity=0.8, visible='legendonly')
        if a.fills is not None:
            f = a.fills
            for side, name, color, sym in ((1, 'Buy fills', BLUE, 'triangle-up'), (-1, 'Sell fills', ORANGE, 'triangle-down')):
                ff = f[f['side'] == side]
                fig.add_trace(go.Scattergl(
                    x=ff['t_s'], y=ff['price'], name=name, mode='markers',
                    marker=dict(color=color, size=6, symbol=sym, opacity=0.7),
                    customdata=np.stack([ff['qty'], ff['inv_after']], axis=1),
                    hovertemplate='%{y:.0f} × %{customdata[0]} → inv %{customdata[1]}<extra>' + name + '</extra>',
                ), row=r, col=1)
    line('Drawdown', g['drawdown'], NEG, row['Drawdown'], fill='tozeroy', fillcolor='rgba(230,103,103,0.15)')
    for name, r in row.items():
        fig.update_yaxes(title_text=name, row=r, col=1)
    fig.update_xaxes(title_text='Time (s)', row=len(rows), col=1)
    fig.update_xaxes(showspikes=True, spikemode='across', spikethickness=1, spikecolor=AXIS)
    return fig


def _per_run_figures(a: Analysis) -> list[str]:
    run = a.run
    out = [_panel(_timeline(a), wide=True, note=(
        'Shared time axis: zoom any row to zoom all. '
        + ('<b>Spread edge</b> = what fills earned vs the mid at the moment they traded; <b>Inventory PnL</b> = mark-to-mid '
           'PnL minus edge (what holding the position afterwards cost or earned — adverse selection lives here). '
           'Fill markers hover with size and resulting inventory; user quotes are toggled from the legend.'
           if run.full_logs else 'Run made with --no-logs: re-run without it for inventory, fills and decomposition.')))]

    b = a.buckets
    fig = figure(f'PnL per {a.bucket_s // 60} min', x='Bucket start (s)', y='PnL')
    fig.add_bar(x=b['start_s'], y=b['pnl'], name='Sim PnL', marker_color=[AQUA if v >= 0 else NEG for v in b['pnl']])
    if 'edge' in b:
        fig.add_scatter(x=b['start_s'], y=b['edge'], name='Spread edge', mode='lines', line=dict(color=INK_2, width=1, dash='dot'))
    out.append(_panel(fig, 'Regime check: is PnL steady or carried by a few windows? Edge (dotted) above the bar means the gap was lost to inventory.'))

    fig = figure('Per-minute PnL distribution', x='PnL change over 1 min', y='Minutes')
    fig.add_histogram(x=a.minute_pnl, nbinsx=60, marker_color=BLUE)
    fig.add_vline(x=0, line=dict(color=AXIS))
    out.append(_panel(fig, f'Mean {fmt(a.summary["minute_mean"])}, std {fmt(a.summary["minute_std"])}. A fat left tail means rare large losses.'))

    if a.markouts is not None:
        fig = figure('Markout by horizon', x='Horizon after fill (s)', y='Ticks per unit (vw)', height=340)
        fig.update_layout(margin=dict(b=90), legend=dict(yanchor='top', y=-0.22, xanchor='left', x=0))
        labels = [f'{h:g}' for h in MARKOUT_HORIZONS_S]
        for grp, color, dash in (('all', INK, None), ('maker', AQUA, None), ('taker', NEG, None), ('buys', BLUE, 'dot'), ('sells', ORANGE, 'dot')):
            m = a.markouts[a.markouts['group'] == grp]
            if len(m) and m['volume'].iloc[0] > 0:
                fig.add_scatter(x=labels[: len(m)], y=m['ticks'], name=f'{grp} ({fmt(m["volume"].iloc[0])} u)', mode='lines+markers',
                                line=dict(color=color, dash=dash, width=1.6), marker=dict(size=5))
        fig.add_hline(y=0, line=dict(color=AXIS))
        fig.update_xaxes(type='category')
        out.append(_panel(fig, 'side·(mid(t+h) − fill price). h=0 is the spread edge captured; the drop from there is adverse '
                               'selection. Flat after a few seconds = the edge is real; still falling = being picked off.'))

    if a.inventory_hist is not None:
        h = a.inventory_hist
        fig = figure('Inventory distribution (time-weighted)', x='Inventory', y='Share of time')
        fig.add_bar(x=h['inventory'], y=h['seconds'] / h['seconds'].sum(), marker_color=BLUE, marker_line_width=0)
        for lim in (run.position_limit, -run.position_limit):
            fig.add_vline(x=lim, line=dict(color=NEG, dash='dot'))
        out.append(_panel(fig, 'Mass piled at ±limit means the strategy sits pinned and quotes one-sided much of the time.'))

    if a.fills is not None and 'edge' in a.fills:
        f = a.fills
        fig = figure('Fill edge vs mid', x='side·(mid before − price), ticks', y='Volume', barmode='group')
        for grp, mask, color in (('maker', f['maker'], AQUA), ('taker', ~f['maker'], NEG)):
            if mask.any():
                e = f.loc[mask].groupby(f.loc[mask, 'edge'].round(2))['qty'].sum()
                fig.add_bar(x=e.index, y=e.values, name=grp, marker_color=color, opacity=0.8)
        out.append(_panel(fig, 'Where fills happened relative to the mid. 0.5 = filled at the touch with a 1-tick spread.'))

        fig = figure('Fill sizes', x='Units per fill', y='Fills')
        fig.add_histogram(x=f['qty'], nbinsx=50, marker_color=BLUE)
        out.append(_panel(fig))

    if 'volume' in b:
        fig = figure(f'Traded volume per {a.bucket_s // 60} min', x='Bucket start (s)', y='Units', barmode='relative')
        fig.add_bar(x=b['start_s'], y=b['buy_volume'], name='Bought', marker_color=BLUE)
        fig.add_bar(x=b['start_s'], y=-b['sell_volume'], name='Sold', marker_color=ORANGE)
        out.append(_panel(fig))

    if a.orders is not None:
        o = a.orders
        counts = o['outcome'].value_counts()
        fig = figure('Order outcomes', y='Orders')
        fig.add_bar(x=counts.index, y=counts.values, marker_color=[SERIES[i % len(SERIES)] for i in range(len(counts))])
        out.append(_panel(fig, f'{fmt(len(o))} orders sent. "cancelled" includes IOC residuals; cancels that lost the race: '
                               f'{fmt(a.summary.get("cancels_failed"))}.'))

        life = o['lifetime_ms'].dropna()
        life = life[life > 0]
        if len(life):
            fig = figure('Order lifetime', x='Accept → fill/cancel (ms, log)', y='Orders', barmode='overlay')
            for outcome, color in (('filled', AQUA), ('cancelled', ORANGE), ('partial+cancelled', YELLOW)):
                v = o.loc[(o['outcome'] == outcome) & (o['lifetime_ms'] > 0), 'lifetime_ms']
                if len(v):
                    fig.add_histogram(x=np.log10(v), name=outcome, marker_color=color, opacity=0.7, nbinsx=60)
            ticks = [1, 10, 100, 1000, 10000, 100000]
            fig.update_xaxes(tickvals=np.log10(ticks), ticktext=[f'{t:g}' for t in ticks])
            out.append(_panel(fig, 'Very short-lived cancels mean quote churn; latency is 5 ms each way for cancels.'))

        if 'placement_ticks' in o:
            fig = figure('Order placement vs mid at send', x='side·(mid − price), ticks (>0 passive)', y='Orders', barmode='group')
            for side, name, color in ((1, 'bids', BLUE), (-1, 'asks', ORANGE)):
                v = o.loc[o['side'] == side, 'placement_ticks'].dropna().round(1)
                if len(v):
                    c = v.clip(-10, 30).value_counts().sort_index()
                    fig.add_bar(x=c.index, y=c.values, name=name, marker_color=color, opacity=0.75)
            out.append(_panel(fig, 'Negative = the order crossed the mid (aggressive). Clipped to [−10, 30].'))

    g = a.grid
    if 'bid_depth' in g:
        fig = figure('User quote distance from market touch', x='Ticks behind best (≤0 at/inside)', y='Share of quoted time', barmode='group')
        for col, name, color in (('bid_depth', 'bid', BLUE), ('ask_depth', 'ask', ORANGE)):
            v = g[col].dropna()
            if len(v):
                c = v.clip(-5, 20).value_counts(normalize=True).sort_index()
                fig.add_bar(x=c.index, y=c.values, name=name, marker_color=color, opacity=0.75)
        out.append(_panel(fig, 'Sampled every second against the market-only book (the user\'s own orders are excluded from it).'))
    return out


def _run_page(a: Analysis) -> str:
    run, s = a.run, a.summary
    m = run.metadata
    meta = (f'<b>dir</b> {html.escape(str(run.log_dir))} · <b>strategy</b> {html.escape(str(m.get("trader_class", "?")))} · '
            f'<b>tape</b> {html.escape(s["tape"])} · <b>duration</b> {s["duration_s"]:,.0f}s · '
            f'<b>latency</b> {float(m.get("latency", 0)) * 1000:g} ms · <b>limit</b> ±{run.position_limit} · '
            f'<b>run</b> {html.escape(str(m.get("run_timestamp", "?")))}')
    warns = ''.join(f'<div class="warn">⚠ {html.escape(w)}</div>' for w in run.warnings)
    return (f'<h2>{html.escape(run.name)}</h2><div class="meta">{meta}</div>{warns}{_tiles(s)}'
            f'<div class="section-title">Charts</div><div class="grid">{"".join(_per_run_figures(a))}</div>')


# ---------------------------------------------------------------- comparison

def _compare_table(analyses: list[Analysis]) -> str:
    kinds = {k: kind for group in TILES.values() for k, _, kind, _ in group}
    labels = {k: label for group in TILES.values() for k, label, _, _ in group}
    head = ''.join(f'<th>{html.escape(a.run.name)}</th>' for a in analyses)
    body = []
    for key in COMPARE_ROWS:
        vals = [a.summary.get(key) for a in analyses]
        nums = [v for v in vals if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v))]
        if not nums:
            continue
        best = None
        if len(nums) > 1 and key in HIGHER_BETTER:
            best = max(nums)
        elif len(nums) > 1 and key in LOWER_BETTER:
            best = min(nums)
        cells = []
        for v in vals:
            cls = 'best' if best is not None and v == best else ('neg' if _is_neg(v, kinds[key]) else '')
            cells.append(f'<td class="{cls}">{_format(v, kinds[key])}</td>')
        body.append(f'<tr><td>{labels[key]}</td>{"".join(cells)}</tr>')
    return f'<div class="table-wrap"><table><tr><th>Metric</th>{head}</tr>{"".join(body)}</table></div>'


def _compare_page(analyses: list[Analysis]) -> str:
    out = []
    fig = figure('PnL', x='Time (s)', y='PnL', height=420, hovermode='x unified')
    for i, a in enumerate(analyses):
        fig.add_scatter(x=a.grid['t_s'], y=a.grid['pnl'], name=a.run.name, mode='lines', line=dict(color=SERIES[i % len(SERIES)], width=1.4))
    out.append(_panel(fig, wide=True))

    fig = figure('Drawdown', x='Time (s)', y='Drawdown', hovermode='x unified')
    for i, a in enumerate(analyses):
        fig.add_scatter(x=a.grid['t_s'], y=a.grid['drawdown'], name=a.run.name, mode='lines', line=dict(color=SERIES[i % len(SERIES)], width=1.2))
    out.append(_panel(fig))

    fig = figure(f'PnL per bucket', x='Bucket start (s)', y='PnL', barmode='group')
    for i, a in enumerate(analyses):
        fig.add_bar(x=a.buckets['start_s'], y=a.buckets['pnl'], name=a.run.name, marker_color=SERIES[i % len(SERIES)])
    out.append(_panel(fig, 'Bucket length per run: 5 min for runs over an hour, else 1 min.'))

    with_mo = [(i, a) for i, a in enumerate(analyses) if a.markouts is not None]
    if with_mo:
        fig = figure('Markout by horizon (all fills)', x='Horizon (s)', y='Ticks per unit (vw)')
        for i, a in with_mo:
            m = a.markouts[a.markouts['group'] == 'all']
            fig.add_scatter(x=[f'{h:g}' for h in m['horizon_s']], y=m['ticks'], name=a.run.name, mode='lines+markers',
                            line=dict(color=SERIES[i % len(SERIES)]))
        fig.update_xaxes(type='category')
        fig.add_hline(y=0, line=dict(color=AXIS))
        out.append(_panel(fig, 'All fills, volume-weighted.'))

    with_inv = [(i, a) for i, a in enumerate(analyses) if a.inventory_hist is not None]
    if with_inv:
        fig = figure('Inventory distribution', x='Inventory', y='Share of time')
        for i, a in with_inv:
            h = a.inventory_hist
            fig.add_scatter(x=h['inventory'], y=h['seconds'] / h['seconds'].sum(), name=a.run.name, mode='lines', line=dict(color=SERIES[i % len(SERIES)], shape='hvh'))
        out.append(_panel(fig))

        fig = figure('Inventory over time', x='Time (s)', y='Inventory', hovermode='x unified')
        for i, a in with_inv:
            fig.add_scattergl(x=a.grid['t_s'], y=a.grid['inventory'], name=a.run.name, mode='lines', line=dict(color=SERIES[i % len(SERIES)], width=1, shape='hv'))
        out.append(_panel(fig, wide=True))

    return (f'<h2>Comparison</h2><div class="meta">Best value per row in green, where "better" is unambiguous. Hover a tile '
            f'label on a run tab for its definition.</div>{_compare_table(analyses)}'
            f'<div class="section-title">Charts</div><div class="grid">{"".join(out)}</div>')


# ---------------------------------------------------------------- entry points

def _json_safe(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return None if math.isnan(float(v)) else float(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def write_report(analyses: list[Analysis], out: Path) -> Path:
    pages, tabs = [], []
    if len(analyses) > 1:
        pages.append(('compare', 'Compare', _compare_page(analyses)))
    for i, a in enumerate(analyses):
        pages.append((f'run{i}', a.run.name, _run_page(a)))
    if len(pages) > 1:
        tabs = [f'<button class="tab" data-page="{pid}" onclick="showTab(\'{pid}\')">{html.escape(label)}</button>' for pid, label, _ in pages]
    title = analyses[0].run.name if len(analyses) == 1 else f'{len(analyses)} runs'
    body = ''.join(f'<div class="page" id="{pid}">{content}</div>' for pid, _, content in pages)
    doc = (f'<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
           f'<title>Results · {html.escape(title)}</title><style>{CSS}</style>'
           f'<script>{plotly.offline.get_plotlyjs()}</script><script>{JS}</script></head>'
           f'<body><div class="app"><h1>Strategy results · {html.escape(title)}</h1>'
           f'<div class="tabs">{"".join(tabs)}</div>{body}</div></body></html>')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc)
    return out


def _discover(root: Path) -> list[Path]:
    return sorted(p.parent for p in root.glob('*/result.json'))


def main() -> None:
    parser = argparse.ArgumentParser(description='Interactive HTML report (metrics, inventory, PnL decomposition, markouts) over simulator runs')
    parser.add_argument('run_dirs', nargs='*', type=Path, help='Simulator --log-dir directories')
    parser.add_argument('--all', action='store_true', help='Include every run directory directly under --logs-root')
    parser.add_argument('--logs-root', type=Path, default=Path('logs'))
    parser.add_argument('--out', type=Path, help='Output HTML (default: <run>/report.html for one run, <logs-root>/results-report.html for several)')
    parser.add_argument('--dt', type=float, default=1.0, help='Time-series sampling step in seconds (default 1)')
    parser.add_argument('--open', action='store_true', help='Open the report in a browser')
    args = parser.parse_args()

    dirs = list(args.run_dirs) + (_discover(args.logs_root) if args.all else [])
    dirs = list(dict.fromkeys(d.resolve() for d in dirs))
    if not dirs:
        parser.error('give run directories or --all')

    analyses = []
    for d in dirs:
        t0 = time.perf_counter()
        a = analyse(load_run(d), dt=args.dt)
        analyses.append(a)
        (d / 'results_summary.json').write_text(json.dumps({k: _json_safe(v) for k, v in a.summary.items()}, indent=2))
        kind = 'full logs' if a.run.full_logs else 'pnl only'
        print(f'{a.run.name}: pnl {fmt(a.summary["pnl"])} ({kind}, {time.perf_counter() - t0:.1f}s)', flush=True)

    out = args.out or (dirs[0] / 'report.html' if len(dirs) == 1 else args.logs_root / 'results-report.html')
    path = write_report(analyses, out)
    print(f'wrote {path} ({path.stat().st_size / 1e6:.1f} MB)')
    if args.open:
        webbrowser.open(path.resolve().as_uri())


if __name__ == '__main__':
    main()
