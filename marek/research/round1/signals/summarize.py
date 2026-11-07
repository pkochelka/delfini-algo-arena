"""Markdown table of every sweep_sig.py run in results.jsonl (last run per label), paired vs 'base'.

    uv run python research/round1/signals/summarize.py
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

rows = {}
for line in Path(__file__).with_name('results.jsonl').read_text().splitlines():
    r = json.loads(line)
    rows[r['label']] = r
base = rows['base']
print('| variant | params | total | delta | fold mean +- sd | t | wins | fills |')
print('| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |')
for name, r in rows.items():
    d = [a - b for a, b in zip(r['pnls'], base['pnls'])]
    m, s = statistics.fmean(d), (statistics.stdev(d) if any(d) else 0.0)
    t = m / (s / len(d) ** 0.5) if s else 0.0
    ps = ', '.join(f'{k}={v}' for k, v in r['params'].items()) or '-'
    print(f"| {name} | {ps} | {r['total']:,.0f} | {sum(d):+,.0f} | {m:+,.0f} +- {s:,.0f} | {t:+.1f} | "
          f"{sum(x > 0 for x in d)}/10 | {r['fills']:,} |")
