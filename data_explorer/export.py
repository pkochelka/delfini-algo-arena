"""Write the data explorer's statistics and charts to disk, so they can be read without running the app.

    uv run data-explorer --data data/sample_v2.csv --export            # whole tape -> data_explorer/sample_v2/
    uv run data-explorer --data data/sample_v2.csv --export --window 0 1800 --dt 0.1
    uv run python -m data_explorer.export --data data/a.csv data/b.csv  # same, plus a compare-a-b/ report

Every tab is rendered exactly as the app renders it, and the component tree is walked,
so the export follows the tabs without any per-chart code. Per tape (or window) it writes:

    README.md     every tile, fitted parameter, table and note, plus a numeric digest of
                  each chart (small charts in full, large ones summarised); the file to read
    summary.json  F.summary and fit_all for the window, machine-readable
    csv/<tab>/    the data behind each chart: one row per point, `trace` names the series; charts over
                  CSV_MAX_ROWS points (price paths, heatmaps, rolling series) are left out unless --all-csv
    report.html   the interactive charts (plotly.js from the CDN); git-ignored, regenerate it
    png/<tab>/    static images, only with --png (needs `kaleido` and Chrome); git-ignored
"""

from __future__ import annotations

import argparse
import shutil
import html as html_lib
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html

from analysis import features as F
from analysis import load_tape
from analysis.fit import fit_all
from analysis.replay import Tape
from data_explorer.tabs import compare

ROOT = Path(__file__).resolve().parent  # reports go to data_explorer/<tape name>/
INLINE_POINTS = 120  # series with at most this many points are written into README.md in full (100 ACF lags fit)
CSV_MAX_ROWS = 2_000  # larger charts get no CSV (README still summarises them), to keep the exports small in git


# ------------------------------------------------------------------ chart data


def _arr(values) -> np.ndarray | None:
    if values is None:
        return None
    a = np.asarray(values)
    return a if a.size else None


def trace_frame(trace) -> pd.DataFrame:
    """The points of one trace as columns x, y (and error_y); histograms are binned, heatmaps melted."""
    kind = trace.type
    if kind == 'heatmap':
        z = np.asarray(trace.z, dtype='float64')
        xs = _arr(trace.x) if _arr(trace.x) is not None else np.arange(z.shape[1])
        ys = _arr(trace.y) if _arr(trace.y) is not None else np.arange(z.shape[0])
        yy, xx = np.meshgrid(ys, xs, indexing='ij')
        df = pd.DataFrame({'x': xx.ravel(), 'y': yy.ravel(), 'z': z.ravel()})
        return df[df['z'].fillna(0) != 0]  # empty cells left out: absent (x, y) means z = 0
    if kind == 'histogram':
        v = np.asarray(trace.x if trace.x is not None else trace.y, dtype='float64')
        v = v[np.isfinite(v)]
        if not len(v):
            return pd.DataFrame({'x': [], 'y': []})
        xb = trace.xbins
        if xb is not None and xb.size:
            edges = np.arange(xb.start, xb.end + xb.size, xb.size)
        else:
            edges = np.histogram_bin_edges(v, bins='auto')[:201]
        counts, edges = np.histogram(v, edges)
        return pd.DataFrame({'x': (edges[:-1] + edges[1:]) / 2, 'y': counts})
    x, y = _arr(getattr(trace, 'x', None)), _arr(getattr(trace, 'y', None))
    if x is None and y is None:
        return pd.DataFrame({'x': [], 'y': []})
    n = len(y) if y is not None else len(x)
    df = pd.DataFrame({'x': x if x is not None else np.arange(n), 'y': y if y is not None else np.full(n, np.nan)})
    err = getattr(trace, 'error_y', None)
    if err is not None and err.array is not None and len(err.array) == n:
        df['error_y'] = np.asarray(err.array)
    return df


def figure_frame(fig: go.Figure) -> pd.DataFrame:
    frames = []
    for i, trace in enumerate(fig.data):
        df = trace_frame(trace)
        fallback = axis_labels(fig)[1] if len(fig.data) == 1 else f'{trace.type} {i + 1}'
        df.insert(0, 'trace', trace.name or fallback)
        if getattr(trace, 'yaxis', None) not in (None, 'y'):
            df.insert(1, 'axis', trace.yaxis)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def figure_title(fig: go.Figure) -> str | None:
    title = fig.layout.title.text if fig.layout.title else None
    if title:
        return title
    subtitles = [a.text for a in fig.layout.annotations if a.text and a.xref == 'paper' and a.yref == 'paper' and a.showarrow is not True]
    return ' / '.join(subtitles) if subtitles else None


def axis_labels(fig: go.Figure) -> tuple[str, str]:
    x = fig.layout.xaxis.title.text if fig.layout.xaxis and fig.layout.xaxis.title else None
    y = fig.layout.yaxis.title.text if fig.layout.yaxis and fig.layout.yaxis.title else None
    return x or 'x', y or 'y'


def _num(v) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not np.isfinite(f):
        return '-'
    if f.is_integer() and abs(f) < 1e12:
        return f'{int(f)}'
    return f'{f:.4g}'


def shapes_digest(fig: go.Figure) -> list[str]:
    """Reference lines and shaded bands: significance bands, zero lines, the latency marker."""
    out = []
    for s in fig.layout.shapes:
        if s.type == 'rect' and s.xref in ('x domain', 'paper'):
            out.append(f'shaded band y ∈ [{_num(s.y0)}, {_num(s.y1)}]')
        elif s.type == 'rect':
            out.append(f'shaded band x ∈ [{_num(s.x0)}, {_num(s.x1)}]')
        elif s.type == 'line' and s.x0 == s.x1 and s.xref not in ('paper', 'x domain'):
            out.append(f'vertical line x = {_num(s.x0)}')
        elif s.type == 'line' and s.y0 == s.y1:
            out.append(f'horizontal line y = {_num(s.y0)}')
    return list(dict.fromkeys(out))


def _wide_table(df: pd.DataFrame, names: list[str], xl: str, yl: str) -> list[str]:
    """Series that share an x axis as one table: a row per x, a column per series."""
    has_err = 'error_y' in df and df['error_y'].notna().any()
    wide = df.pivot_table(index='x', columns='trace', values='y', sort=False, aggfunc='first')
    err = df.pivot_table(index='x', columns='trace', values='error_y', sort=False, aggfunc='first') if has_err else None
    cols = [n for n in names if n in wide.columns]
    lines = [f'| {xl} | ' + ' | '.join(str(c) for c in cols) + ' |', '|' + ' --- |' * (len(cols) + 1)]
    for x, row in wide.iterrows():
        cells = []
        for c in cols:
            cell = _num(row[c])
            if err is not None and c in err.columns and pd.notna(err.at[x, c]):
                cell += f' ± {_num(1.96 * err.at[x, c])}'
            cells.append(cell)
        lines.append(f'| {_num(x)} | ' + ' | '.join(cells) + ' |')
    lines += ['', f'y: {yl}' + ('  (± is the 95% interval)' if has_err else '')]
    return lines


def figure_digest(fig: go.Figure, df: pd.DataFrame) -> str:
    """Markdown numbers for one chart: the whole series when small, a per-series summary when large."""
    if df.empty:
        texts = [a.text for a in fig.layout.annotations if a.text]
        return f'_No data: {"; ".join(texts)}_' if texts else '_No data._'
    xl, yl = axis_labels(fig)
    lines = []
    sizes = df.groupby('trace', sort=False).size()
    if 'z' in df:
        z = df['z'].to_numpy(dtype='float64')
        lines.append(f'Heatmap, {len(df):,} non-zero cells (missing cells in the CSV are 0); x ({xl}) {_num(df["x"].min())}..{_num(df["x"].max())}, y ({yl}) {_num(df["y"].min())}..{_num(df["y"].max())}; z mean {_num(np.nanmean(z))}, max {_num(np.nanmax(z))}.')
    else:
        small = [t for t in sizes.index if sizes[t] <= INLINE_POINTS]
        large = [t for t in sizes.index if sizes[t] > INLINE_POINTS]
        if small:
            lines += _wide_table(df[df['trace'].isin(small)], small, xl, yl)
        if large:
            if small:
                lines.append('')
            lines.append(f'| series | points | {xl} range | {yl} min | mean | median | max | last |')
            lines.append('| --- | --- | --- | --- | --- | --- | --- | --- |')
            for name in large:
                g = df[df['trace'] == name]
                y = pd.to_numeric(g['y'], errors='coerce')
                lines.append(
                    f'| {name} | {len(g):,} | {_num(g["x"].min())}..{_num(g["x"].max())} | {_num(y.min())} | {_num(y.mean())} | {_num(y.median())} | {_num(y.max())} | {_num(y.iloc[-1])} |'
                )
    shapes = shapes_digest(fig)
    if shapes:
        lines += ['', 'Marked: ' + '; '.join(shapes) + '.']
    return '\n'.join(lines)


# ------------------------------------------------------------ component walk


def text_of(node) -> str:
    if node is None:
        return ''
    if isinstance(node, (str, int, float)):
        return str(node)
    if isinstance(node, (list, tuple)):
        return ' '.join(t for t in (text_of(c) for c in node) if t)
    return text_of(getattr(node, 'children', None))


def _classes(node) -> set[str]:
    return set((getattr(node, 'className', None) or '').split())


def _md_cell(s: str) -> str:
    return s.replace('|', '\\|').replace('\n', ' ')


@dataclass
class Report:
    """Collects one report as markdown and HTML while the component tree is walked."""

    out_dir: Path
    png: bool = False
    all_csv: bool = False
    md: list[str] = field(default_factory=list)
    html: list[str] = field(default_factory=list)
    tab_key: str = ''
    figures: int = 0
    plotly_js_done: bool = False

    def heading(self, level: int, text: str) -> None:
        self.md += [f'{"#" * level} {text}', '']
        self.html.append(f'<h{level}>{html_lib.escape(text)}</h{level}>')

    def para(self, text: str, *, note: bool = False) -> None:
        if not text.strip():
            return
        self.md += [f'> {text}' if note else text, '']
        self.html.append(f'<p class="{"note" if note else ""}">{html_lib.escape(text)}</p>')

    def code(self, text: str) -> None:
        self.md += ['```python', text, '```', '']
        self.html.append(f'<pre>{html_lib.escape(text)}</pre>')

    def table(self, rows: list[list[str]]) -> None:
        if not rows:
            return
        width = max(len(r) for r in rows)
        rows = [r + [''] * (width - len(r)) for r in rows]
        self.md.append('| ' + ' | '.join(_md_cell(c) for c in rows[0]) + ' |')
        self.md.append('|' + ' --- |' * width)
        self.md += ['| ' + ' | '.join(_md_cell(c) for c in r) + ' |' for r in rows[1:]]
        self.md.append('')
        body = ''.join('<tr>' + ''.join(f'<td>{html_lib.escape(c)}</td>' for c in r) + '</tr>' for r in rows)
        self.html.append(f'<table>{body}</table>')

    def figure(self, fig: go.Figure) -> None:
        self.figures += 1
        title = figure_title(fig) or f'Chart {self.figures}'
        slug = re.sub(r'[^a-z0-9]+', '-', title.lower()).strip('-')[:60] or 'chart'
        stem = f'{self.figures:02d}-{slug}'
        df = figure_frame(fig)
        if self.all_csv or len(df) <= CSV_MAX_ROWS:
            csv_path = self.out_dir / 'csv' / self.tab_key / f'{stem}.csv'
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(csv_path, index=False, float_format='%.6g')
            rel_csv = csv_path.relative_to(self.out_dir).as_posix()
            links = f'data: [`{rel_csv}`]({rel_csv})'
        else:
            links = f'data: not saved ({len(df):,} points; `--export --all-csv` writes it)'
        if self.png:
            png_path = self.out_dir / 'png' / self.tab_key / f'{stem}.png'
            png_path.parent.mkdir(parents=True, exist_ok=True)
            fig.write_image(png_path, width=1100, height=fig.layout.height or 400, scale=1)
            rel_png = png_path.relative_to(self.out_dir).as_posix()
            links += f' · image: [`{rel_png}`]({rel_png})'
        self.md += [f'#### {title}', '', links, '', figure_digest(fig, df), '']
        include = 'cdn' if not self.plotly_js_done else False
        self.plotly_js_done = True
        self.html.append('<div class="chart">' + fig.to_html(include_plotlyjs=include, full_html=False, config={'displaylogo': False}) + '</div>')

    def walk(self, node) -> None:
        if node is None:
            return
        if isinstance(node, (str, int, float)):
            self.para(str(node))
            return
        if isinstance(node, (list, tuple)):
            for child in node:
                self.walk(child)
            return
        if isinstance(node, dcc.Graph):
            fig = node.figure
            self.figure(fig if isinstance(fig, go.Figure) else go.Figure(fig))
            return
        cls = _classes(node)
        if isinstance(node, html.Table):
            self.table([[text_of(cell) for cell in (row.children or [])] for row in (node.children or [])])
        elif 'tiles' in cls:
            rows = [['Statistic', 'Value', 'Detail']]
            for tile in node.children or []:
                parts = [text_of(c) for c in tile.children or []]
                rows.append((parts + ['', '', ''])[:3])
            self.table(rows)
        elif 'code' in cls:
            self.code(text_of(node))
        elif isinstance(node, (html.H1, html.H2, html.H3, html.H4)):
            self.heading(3, text_of(node))
        elif isinstance(node, html.P):
            self.para(text_of(node), note='note' in cls)
        elif isinstance(getattr(node, 'children', None), str):
            self.para(node.children, note='note' in cls)
        else:
            self.walk(getattr(node, 'children', None))

    def write(self, title: str, preamble: list[str]) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / 'README.md').write_text('\n'.join([f'# {title}', '', *preamble, '', *self.md]).rstrip() + '\n')
        page = (
            f'<!doctype html><html><head><meta charset="utf-8"><title>{html_lib.escape(title)}</title><style>'
            'body{background:#0d0d0d;color:#f0efec;font-family:system-ui,sans-serif;max-width:1200px;margin:0 auto;padding:16px}'
            'h2{border-bottom:1px solid #383835;padding-bottom:4px;margin-top:40px}h3{color:#c3c2b7}'
            '.note{color:#898781;font-size:13px;margin:4px 0 20px}pre{background:#1a1a19;padding:12px;overflow-x:auto}'
            'table{border-collapse:collapse;margin:8px 0 20px}td{border:1px solid #2c2c2a;padding:4px 8px;font-size:13px}'
            '.chart{margin:12px 0 0}</style></head><body>'
            f'<h1>{html_lib.escape(title)}</h1>' + ''.join(f'<p>{html_lib.escape(p)}</p>' for p in preamble if p) + ''.join(self.html) + '</body></html>'
        )
        (self.out_dir / 'report.html').write_text(page)


# ----------------------------------------------------------------- exporting


def _tabs():
    from data_explorer.main import TABS  # late import: main imports this module for --export

    return [(key, label, render) for key, (label, render) in TABS.items() if render is not None]


def out_dir_for(tape: Tape, window: F.Window, root: Path = ROOT) -> Path:
    base = root / tape.name
    return base if window is None else base / f'window-{window[0]:g}-{window[1]:g}s'


def export_tape(tape: Tape, window: F.Window = None, dt: float = 1.0, *, root: Path = ROOT, png: bool = False, all_csv: bool = False, source: Path | None = None) -> Path:
    out = out_dir_for(tape, window, root)
    _clear(out)
    report = Report(out, png=png, all_csv=all_csv)
    t0, t1 = F.bounds(tape, window)
    started = time.perf_counter()
    for key, label, render in _tabs():
        tab_started = time.perf_counter()
        report.tab_key = key
        report.figures = 0
        report.heading(2, label)
        report.walk(render(tape, window, dt))
        print(f'  {key}: {report.figures} charts in {time.perf_counter() - tab_started:.1f}s', flush=True)

    summary = {'tape': tape.name, 'window_s': [t0, t1], 'dt_s': dt, 'summary': F.summary(tape, window), 'fit': fit_all(tape, window)}
    (out / 'summary.json').parent.mkdir(parents=True, exist_ok=True)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, default=_json_default) + '\n')

    preamble = [
        f'Data explorer export of `{source.as_posix() if source else tape.name}`, '
        f'window {t0:,.1f}–{t1:,.1f} s ({(t1 - t0) / 60:.1f} min), sampling selector {dt:g} s '
        f'(charts that use it: return histogram and ACF, rolling σ, variance ratio, OFI scatter).',
        f'Generated {time.strftime("%Y-%m-%d %H:%M")} by `uv run data-explorer --data … --export` (see data_explorer/export.py). '
        'Numbers are in ticks and seconds. The data behind each chart is in csv/, machine-readable headline stats in summary.json, '
        'interactive charts in report.html.',
        '',
        'Contents: ' + ' · '.join(f'[{label}](#{re.sub(r"[^a-z0-9 -]", "", label.lower()).replace(" ", "-")})' for _, label, _ in _tabs()),
    ]
    report.write(f'Data explorer: {tape.name}' + (f', {t0:g}–{t1:g} s' if window else ''), preamble)
    print(f'Wrote {out} in {time.perf_counter() - started:.1f}s', flush=True)
    return out


def export_compare(tapes: list[Tape], *, root: Path = ROOT, png: bool = False, all_csv: bool = False) -> Path:
    out = root / ('compare-' + '-'.join(t.name for t in tapes))
    _clear(out)
    report = Report(out, png=png, all_csv=all_csv)
    report.tab_key = 'compare'
    report.walk(compare.render(tapes))
    report.write('Data explorer: ' + ' vs '.join(t.name for t in tapes), [f'Whole tapes side by side, generated {time.strftime("%Y-%m-%d %H:%M")}.'])
    print(f'Wrote {out}', flush=True)
    return out


def _clear(out: Path) -> None:
    """Drop the previous export's chart files, so renamed or no-longer-saved charts do not linger."""
    for sub in ('csv', 'png'):
        shutil.rmtree(out / sub, ignore_errors=True)


def _json_default(v):
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    return str(v)


def run(paths: list[Path], window: F.Window, dt: float, *, refresh: bool = False, png: bool = False, all_csv: bool = False) -> None:
    tapes = []
    for path in paths:
        tape = load_tape(path, refresh=refresh)
        tapes.append(tape)
        export_tape(tape, window, dt, png=png, all_csv=all_csv, source=path)
    if len(tapes) > 1:
        export_compare(tapes, png=png, all_csv=all_csv)


def main() -> None:
    parser = argparse.ArgumentParser(description='Export the data explorer statistics and charts to data_explorer/<tape>/')
    parser.add_argument('--data', type=Path, nargs='+', required=True)
    parser.add_argument('--window', type=float, nargs=2, metavar=('T0', 'T1'), help='seconds; default the whole tape')
    parser.add_argument('--dt', type=float, default=1.0, help='the sampling selector, s (default 1)')
    parser.add_argument('--png', action='store_true', help='also write PNGs (needs kaleido and Chrome)')
    parser.add_argument('--all-csv', action='store_true', help=f'also write CSVs of charts over {CSV_MAX_ROWS:,} points')
    parser.add_argument('--refresh', action='store_true', help='rebuild the replay cache')
    args = parser.parse_args()
    run(args.data, tuple(args.window) if args.window else None, args.dt, refresh=args.refresh, png=args.png, all_csv=args.all_csv)


if __name__ == '__main__':
    main()
