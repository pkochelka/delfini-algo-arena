"""Cut a tape into regime-pure stretches (calm / turbulent), one CSV per stretch, for per-regime evaluation.

Usage (repo root):
    uv run python strategies/peter/regime_split.py data/round2_public.csv --out-dir data/splits_r2/regime --test-from 4500

Labels are offline (they use the future); the strategies must detect regimes causally.
    jump      |mid change| >= --jump ticks within one 100 ms bin
    outage    maker insert rate (1 s centred mean) < --outage-rate /s: the ~3 s maker silence that precedes
              the big round-2 jumps
    turbulent union of [outage start - lead, outage end + tail] and [jump - lead, jump + tail], merged when
              closer than --merge s.  Starting `lead` s before the outage means the strategy enters each
              turbulent stretch in a calm-looking state and has to detect it itself.
    calm      the rest, minus --margin s on each side, cut into pieces of at most --calm-max s; pieces
              shorter than --calm-min are dropped.
Each file holds [start - warmup, end) with the resting book seeded at its first timestamp (split_data.py);
regime.json lists name, regime, set (dev / test by --test-from), trade_start_us and the window.
"""
from __future__ import annotations

import argparse, csv, json, sys
from pathlib import Path

import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1]))
from split_data import HEADER, snapshots_at, write_window  # noqa: E402
from analysis.cache import load_tape  # noqa: E402


def labels(tape_csv: str, jump: float, outage_rate: float):
    t = load_tape(tape_csv, verbose=False); ev, o = t.events, t.orders
    T = ev.t_us.to_numpy(); dur = T[-1] / 1e6
    bid, ask = ev.bid.to_numpy().astype(float), ev.ask.to_numpy().astype(float); two = (bid > 0) & (ask > 0)
    MID = pd.Series(np.where(two, (bid + ask) / 2, np.nan)).ffill().bfill().to_numpy()
    nb = int(dur * 10); g = (np.arange(nb) + 1) * 100_000
    m01 = MID[np.searchsorted(T, g, 'right') - 1]
    J = (np.flatnonzero(np.abs(np.diff(m01)) >= jump) + 1) / 10.0
    gm = o[~o.ioc]
    mk = np.bincount((gm.t_us.to_numpy() // 100_000).astype(int), minlength=nb)[:nb].astype(float)
    sm = pd.Series(mk).rolling(10, center=True, min_periods=5).mean().to_numpy() * 10
    out = sm < outage_rate
    st = np.flatnonzero(out & ~np.r_[False, out[:-1]]); en = np.flatnonzero(out & ~np.r_[out[1:], False])
    outages = [(a / 10, (b + 1) / 10) for a, b in zip(st, en) if (b - a) >= 10]
    return dur, J, outages


def merge(iv, gap):
    iv = sorted(iv); out = []
    for a, b in iv:
        if out and a <= out[-1][1] + gap:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('csv'); ap.add_argument('--out-dir', type=Path, default=Path('data/splits_r2/regime'))
    ap.add_argument('--jump', type=float, default=5); ap.add_argument('--outage-rate', type=float, default=110)
    ap.add_argument('--lead', type=float, default=5); ap.add_argument('--tail', type=float, default=30)
    ap.add_argument('--merge', type=float, default=30); ap.add_argument('--margin', type=float, default=5)
    ap.add_argument('--calm-min', type=float, default=30); ap.add_argument('--calm-max', type=float, default=300)
    ap.add_argument('--warmup', type=float, default=30); ap.add_argument('--test-from', type=float, default=4500)
    a = ap.parse_args()
    dur, J, outages = labels(a.csv, a.jump, a.outage_rate)
    iv = [(s - a.lead, e + a.tail) for s, e in outages] + [(j - a.lead, j + a.tail) for j in J]
    turb = [[max(a.warmup, s), min(dur, e)] for s, e in merge(iv, a.merge)]
    calm, prev = [], a.warmup
    for s, e in turb + [[dur + a.margin, dur]]:
        lo, hi = prev + a.margin, s - a.margin
        while hi - lo >= a.calm_min:
            n = int(np.ceil((hi - lo) / a.calm_max)); step = (hi - lo) / n
            calm += [[lo + k * step, lo + (k + 1) * step] for k in range(n)]; break
        prev = e
    with open(a.csv, newline='') as f:
        r = csv.reader(f); assert next(r) == HEADER; rows = list(r)
    times = [float(x[0]) for x in rows]
    a.out_dir.mkdir(parents=True, exist_ok=True)
    items = [('turb', s, e) for s, e in turb] + [('calm', s, e) for s, e in calm]
    items.sort(key=lambda x: x[1])
    starts = [max(0.0, s - a.warmup) for _, s, _ in items]
    order = np.argsort(starts); snaps = [None] * len(items)
    for k, snap in zip(order, snapshots_at(rows, [starts[k] for k in order])):
        snaps[k] = snap
    meta, cnt = [], {'calm': 0, 'turb': 0}
    for (reg, s, e), st, snap in zip(items, starts, snaps):
        cnt[reg] += 1; name = f'{reg}{cnt[reg]:03d}'
        n = write_window(a.out_dir / f'{name}.csv', rows, times, st, e, snap)
        meta.append(dict(name=name, file=f'{name}.csv', regime=reg, set='test' if s >= a.test_from else 'dev',
                         start_s=st, eval_start_s=s, end_s=e, trade_start_us=int(round(s * 1e6)), rows=n))
    (a.out_dir / 'regime.json').write_text(json.dumps({'source': a.csv, 'args': {k: str(v) for k, v in vars(a).items()},
                                                       'outages': outages, 'jumps': J.tolist(), 'stretches': meta}, indent=1))
    D = pd.DataFrame(meta); D['len'] = D.end_s - D.eval_start_s
    print(D.groupby(['regime', 'set'])['len'].agg(['count', 'sum', 'median', 'max']).round(1).to_string())
    print(f'outages {len(outages)}, jumps {len(J)}; turbulent share of tape {sum(e - s for s, e in turb) / dur:.3f}')


if __name__ == '__main__':
    main()
