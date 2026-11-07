"""Self-contained HTML report for a walk-forward run (reads only the files `walk-forward` wrote).

    uv run walk-forward-report --run-dir logs/wf/<run> [--open]
    uv run walk-forward-report --run-dir logs/wf/a logs/wf/b     # one report per run

Styling follows data_explorer/theme.py (dark, the same validated palette).
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import webbrowser
from pathlib import Path

import plotly.graph_objects as go
import plotly.offline
from plotly.subplots import make_subplots

from data_explorer.theme import AQUA, AXIS, BLUE, GRID, INK, INK_2, MUTED, ORANGE, PAGE, SURFACE, figure, fmt

NEG = '#e66767'  # the negative pole of data_explorer's diverging pair

CSS = f"""
body {{ margin:0; background:{PAGE}; color:{INK}; font-family:system-ui,-apple-system,"Segoe UI",sans-serif; }}
.app {{ padding:16px; max-width:1500px; margin:0 auto; }}
h1 {{ font-size:18px; font-weight:600; margin:0 0 4px; }}
.meta {{ font-size:12px; color:{MUTED}; margin-bottom:14px; line-height:1.6; word-break:break-all; }}
.meta b {{ color:{INK_2}; font-weight:500; }}
.section-title {{ font-size:13px; font-weight:600; color:{INK_2}; text-transform:uppercase; letter-spacing:.04em; margin:18px 2px 8px; }}
.tiles {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(150px,1fr)); gap:8px; }}
.tile {{ background:{SURFACE}; border-radius:8px; padding:10px 12px; }}
.tile-label {{ font-size:11px; color:{MUTED}; text-transform:uppercase; letter-spacing:.04em; }}
.tile-value {{ font-size:20px; font-weight:600; margin-top:4px; font-variant-numeric:tabular-nums; }}
.tile-sub {{ font-size:11px; color:{MUTED}; margin-top:2px; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(min(520px,100%),1fr)); gap:12px; }}
.panel {{ background:{SURFACE}; border-radius:8px; padding:4px 4px 8px; min-width:0; }}
.panel.wide {{ grid-column:1/-1; }}
.note {{ font-size:12px; color:{MUTED}; margin:2px 12px 4px; line-height:1.45; }}
.table-wrap {{ overflow-x:auto; background:{SURFACE}; border-radius:8px; }}
table {{ border-collapse:collapse; width:100%; font-size:12px; font-variant-numeric:tabular-nums; }}
th, td {{ padding:6px 10px; text-align:right; border-bottom:1px solid {GRID}; white-space:nowrap; }}
th {{ color:{MUTED}; font-weight:500; position:sticky; top:0; background:{SURFACE}; }}
td:first-child, th:first-child {{ text-align:left; }}
td.pos {{ color:{INK}; }} td.neg {{ color:{NEG}; }}
"""


def _load(run_dir: Path) -> tuple[dict, dict, list[dict], dict[str, list[tuple[float, float]]]]:
    config = json.loads((run_dir / 'config.json').read_text())
    summary = json.loads((run_dir / 'summary.json').read_text())
    with (run_dir / 'folds.csv').open() as f:
        rows = [{k: (float(v) if v not in ('', 'None') and k != 'fold' else v) for k, v in r.items()}
                for r in csv.DictReader(f)]
    curves = {}
    for r in rows:
        with (run_dir / 'pnl' / f'{r["fold"]}.csv').open() as f:
            reader = csv.reader(f)
            next(reader)
            curves[r['fold']] = [(float(t), float(p)) for t, p in reader]
    return config, summary, rows, curves


def _tile(label: str, value, sub: str | None = None) -> str:
    v = value if isinstance(value, str) else fmt(value, 4)
    s = f'<div class="tile-sub">{html.escape(sub)}</div>' if sub else ''
    return f'<div class="tile"><div class="tile-label">{html.escape(label)}</div><div class="tile-value">{v}</div>{s}</div>'


def _div(fig: go.Figure) -> str:
    return plotly.offline.plot(fig, output_type='div', include_plotlyjs=False,
                               config={'displaylogo': False, 'responsive': True})


def _panel(fig: go.Figure, note: str | None = None, wide: bool = False) -> str:
    n = f'<p class="note">{html.escape(note)}</p>' if note else ''
    return f'<div class="panel{" wide" if wide else ""}">{_div(fig)}{n}</div>'


def _figures(rows: list[dict], curves: dict) -> list[str]:
    names = [r['fold'] for r in rows]
    pnls = [r['eval_pnl'] for r in rows]
    windows = [f'{r["eval_start_s"]:.0f}–{r["end_s"]:.0f}s' for r in rows]
    out = []

    # 1. Stitched equity curve: eval windows chained end to end.
    fig = figure('Stitched out-of-sample PnL', x='tape time (s)', y='cumulative PnL', height=340)
    xs, ys, offset = [], [], 0.0
    for r in rows:
        for t, v in curves[r['fold']]:
            xs.append(t)
            ys.append(offset + v)
        offset += r['eval_pnl']
        fig.add_vline(x=r['end_s'], line=dict(color=AXIS, width=1, dash='dot'))
    fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines', line=dict(color=BLUE, width=2), name='cumulative PnL',
                             hovertemplate='t=%{x:.0f}s<br>PnL %{y:,.0f}<extra></extra>'))
    fig.update_layout(hovermode='x unified', showlegend=False)
    out.append(_panel(fig, 'Each fold is a fresh replay; its eval-window PnL is appended to the previous folds\' total. '
                           'Dotted lines are fold boundaries.', wide=True))

    # 2. PnL per fold (polarity: blue gain, red loss).
    fig = figure('PnL per fold', y='eval-window PnL', height=320)
    fig.add_trace(go.Bar(x=names, y=pnls, marker=dict(color=[BLUE if p >= 0 else NEG for p in pnls], line=dict(width=0)),
                         customdata=windows, hovertemplate='%{x} (%{customdata})<br>PnL %{y:,.2f}<extra></extra>'))
    mean = sum(pnls) / len(pnls)
    fig.add_hline(y=mean, line=dict(color=INK_2, width=1, dash='dash'),
                  annotation=dict(text=f'mean {fmt(mean, 4)}', font=dict(color=INK_2, size=11)), annotation_position='top left')
    fig.update_layout(showlegend=False, bargap=0.25)
    out.append(_panel(fig, 'Consistency matters more than the total: one huge fold carrying the rest is a warning sign.'))

    # 3. Drawdown per fold next to PnL magnitude.
    fig = figure('Max drawdown inside each fold', y='drawdown', height=320)
    fig.add_trace(go.Bar(x=names, y=[r['max_drawdown'] for r in rows], marker=dict(color=ORANGE, line=dict(width=0)),
                         hovertemplate='%{x}<br>max drawdown %{y:,.2f}<extra></extra>'))
    fig.update_layout(showlegend=False, bargap=0.25)
    out.append(_panel(fig, 'Peak-to-trough fall of the eval-window PnL curve.'))

    # 4. Inventory quantiles per fold (time-weighted |inventory|).
    fig = figure('|Inventory| quantiles per fold', y='lots', height=320)
    for key, label, color in (('abs_inventory_q50', 'q50', BLUE), ('abs_inventory_q95', 'q95', ORANGE),
                              ('abs_inventory_q997', 'q99.7', AQUA)):
        fig.add_trace(go.Bar(x=names, y=[r[key] for r in rows], name=label, marker=dict(color=color, line=dict(width=0)),
                             hovertemplate=f'%{{x}} {label}: %{{y}}<extra></extra>'))
    fig.add_hline(y=100, line=dict(color=NEG, width=1, dash='dash'),
                  annotation=dict(text='penalty limit 100', font=dict(color=INK_2, size=11)), annotation_position='top left')
    fig.update_layout(barmode='group', bargap=0.2, bargroupgap=0.08)
    out.append(_panel(fig, 'Time-weighted over the whole fold file (warmup included). Above 100 costs $10k/s.'))

    # 5. Activity per fold.
    fig = figure('Fills per fold', y='fills', height=320)
    fig.add_trace(go.Bar(x=names, y=[r['fills'] for r in rows], marker=dict(color=AQUA, line=dict(width=0)),
                         customdata=[r['traded_volume'] for r in rows],
                         hovertemplate='%{x}<br>fills %{y:,}<br>volume %{customdata:,}<extra></extra>'))
    fig.update_layout(showlegend=False, bargap=0.25)
    out.append(_panel(fig))

    # 6. Small multiples: each fold's eval-window curve on a shared y-scale.
    n = len(rows)
    cols = min(5, n)
    nrows = math.ceil(n / cols)
    fig = make_subplots(rows=nrows, cols=cols, shared_yaxes=True, subplot_titles=names,
                        horizontal_spacing=0.02, vertical_spacing=0.12 if nrows > 1 else 0.0)
    base = figure(height=170 * nrows + 40)
    fig.update_layout(base.layout)
    fig.update_layout(showlegend=False, title=dict(text='PnL curve inside each fold', font=dict(size=13, color=INK), x=0.01),
                      margin=dict(l=56, r=16, t=60, b=30))
    for i, r in enumerate(rows):
        pts = curves[r['fold']]
        color = BLUE if r['eval_pnl'] >= 0 else NEG
        fig.add_trace(go.Scatter(x=[(t - r['eval_start_s']) for t, _ in pts], y=[v for _, v in pts], mode='lines',
                                 line=dict(color=color, width=1.5), name=r['fold'],
                                 hovertemplate=r['fold'] + ' +%{x:.0f}s: %{y:,.0f}<extra></extra>'),
                      row=i // cols + 1, col=i % cols + 1)
    fig.update_xaxes(gridcolor=GRID, linecolor=AXIS, zerolinecolor=AXIS, tickfont=dict(color=MUTED, size=10))
    fig.update_yaxes(gridcolor=GRID, linecolor=AXIS, zerolinecolor=AXIS, tickfont=dict(color=MUTED, size=10))
    fig.update_annotations(font=dict(size=11, color=INK_2))
    out.append(_panel(fig, 'Seconds since the eval window opened; shared y-scale so folds compare directly.', wide=True))
    return out


def _table(rows: list[dict]) -> str:
    cols = [('fold', 'fold'), ('eval_start_s', 'eval start s'), ('end_s', 'end s'), ('eval_pnl', 'PnL'),
            ('warmup_pnl', 'warmup PnL'), ('max_drawdown', 'max DD'), ('fills', 'fills'), ('traded_volume', 'volume'),
            ('end_inventory', 'end inv'), ('breach_penalty', 'breach'), ('abs_inventory_q50', '|inv| q50'),
            ('abs_inventory_q95', '|inv| q95'), ('abs_inventory_q997', '|inv| q99.7'), ('runtime_s', 'runtime s')]
    head = ''.join(f'<th>{h}</th>' for _, h in cols)
    body = []
    for r in rows:
        cells = []
        for k, _ in cols:
            v = r[k]
            if k == 'fold':
                cells.append(f'<td>{html.escape(v)}</td>')
            else:
                cls = (' class="neg"' if v < 0 else ' class="pos"') if k == 'eval_pnl' else ''
                cells.append(f'<td{cls}>{fmt(round(v, 2), 6)}</td>')
        body.append('<tr>' + ''.join(cells) + '</tr>')
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def write_report(run_dir: Path) -> Path:
    config, s, rows, curves = _load(run_dir)
    sharpe = s.get('fold_sharpe')
    tiles = [
        _tile('Total OOS PnL', s['total_pnl'], f'{s["folds"]} folds'),
        _tile('Mean / fold', s['mean_pnl'], f'median {fmt(s["median_pnl"], 4)}'),
        _tile('Std / fold', s['std_pnl']),
        _tile('Fold Sharpe', '-' if sharpe is None else f'{sharpe:.2f}', 'mean / std, not annualised'),
        _tile('t-stat', '-' if s.get('t_stat') is None else f'{s["t_stat"]:.2f}', '|t| > 2 ≈ distinguishable from 0'),
        _tile('Win rate', f'{s["win_rate"]:.0%}', f'{round(s["win_rate"] * s["folds"])}/{s["folds"]} folds > 0'),
        _tile('Worst fold', s['min_pnl'], f'best {fmt(s["max_pnl"], 4)}'),
        _tile('PnL / hour', s.get('pnl_per_hour')),
        _tile('Max drawdown', s['stitched_max_drawdown'], 'stitched curve'),
        _tile('Breach penalty', s['total_breach_penalty'], 'all folds'),
        _tile('|Inv| q95', s['mean_abs_inventory_q95'], f'mean over folds; max q99.7 {fmt(s["max_abs_inventory_q997"])}'),
        _tile('Fills', s['total_fills'], f'volume {fmt(s["total_volume"])}'),
    ]
    params = json.dumps(config['params']) if config.get('params') else 'defaults'
    meta = (f'<b>strategy</b> {html.escape(config["strategy"])}<br><b>backtest</b> {html.escape(config["backtest"])}<br>'
            f'<b>folds</b> {config["folds"]} · <b>warmup</b> {config["warmup_s"]:g}s · <b>end-frac</b> {config["end_frac"]:g} · '
            f'<b>params</b> {html.escape(params)} · <b>run</b> {config["run_timestamp"]} ({config["wall_time_s"]:.0f}s)')
    title = f'{config["strategy_name"]} · {Path(config["backtest"]).stem} walk-forward'
    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>{CSS}</style>
<script>{plotly.offline.get_plotlyjs()}</script></head>
<body><div class="app">
<h1>{html.escape(title)}</h1><div class="meta">{meta}</div>
<div class="tiles">{''.join(tiles)}</div>
<div class="section-title">Charts</div>
<div class="grid">{''.join(_figures(rows, curves))}</div>
<div class="section-title">Folds</div>
{_table(rows)}
</div></body></html>"""
    path = run_dir / 'report.html'
    path.write_text(doc)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run-dir', type=Path, nargs='+', required=True, help='directory written by walk-forward')
    ap.add_argument('--open', action='store_true', help='open the report in the browser')
    args = ap.parse_args()
    for d in args.run_dir:
        path = write_report(d)
        print(path)
        if args.open:
            webbrowser.open(path.resolve().as_uri())


if __name__ == '__main__':
    main()
