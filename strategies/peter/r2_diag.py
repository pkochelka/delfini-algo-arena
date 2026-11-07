"""Fill diagnostics for r2_mm: where does the PnL come from?

uv run python strategies/peter/r2_diag.py '{"turb_rv":2,"turb_cap":0}' [--folds 1-12] [--dir data/splits_r2/wf]

Runs each fold with PARAMS log=1, then marks every fill out against the tape mid (replayed from the fold CSV)
at +0.1, 1, 10 and 60 s. Rows: passive / take x calm / turbulent x time since the last detected jump.
Markout = sign * (mid(t + h) - price) * qty, i.e. positive = good for us. The 60 s column approximates the
fill's contribution including inventory carry.
"""
from __future__ import annotations

import argparse, json, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'strategies/peter'))
from tune import load_folds, parse_range  # noqa: E402

HS = (0.1, 1.0, 10.0, 60.0)


def run(strategy: str, ov: dict, fold: dict):
    from arena import load_agent
    from simulator.main import MAX_POSITION, MAX_POTENTIAL_POSITION, ORDER_ACCEPT_DELAY_MS, POSITION_BREACH_PENALTY_PER_SECOND
    from simulator.simulator import ExchangeSimulator, SimulationConfig
    from analysis.cache import load_tape
    st = load_agent(Path(strategy), ov | {'log': 1, 'trade_start_us': fold['trade_start_us']}, 'diag')
    cfg = SimulationConfig(max_position=MAX_POSITION, max_potential_position=MAX_POTENTIAL_POSITION,
                           position_breach_penalty_per_second=POSITION_BREACH_PENALTY_PER_SECOND,
                           order_accept_delay_ms=ORDER_ACCEPT_DELAY_MS)
    r = ExchangeSimulator(strategy=st, config=cfg).run(Path(fold['path']))
    ev = load_tape(fold['path'], verbose=False).events
    two = (ev.bid > 0) & (ev.ask > 0)
    T = ev.t_us.to_numpy(); mid = np.where(two, (ev.bid + ev.ask) / 2, np.nan)
    mid = pd.Series(mid).ffill().bfill().to_numpy()
    f = pd.DataFrame(st.fills, columns=['t', 'sg', 'px', 'q', 'take', 'S', 'sj', 'turb'])
    for h in HS:
        i = np.minimum(np.searchsorted(T, f.t.to_numpy() + h * 1e6, 'right') - 1, len(T) - 1)
        f[f'm{h:g}'] = f.sg * (mid[i] - f.px) * f.q
    f['fold'] = fold['name']
    return f, r.pnl, dict(st.stats)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('params', nargs='?', default='{}')
    ap.add_argument('--strategy', default='strategies/peter/r2_mm.py')
    ap.add_argument('--folds', default='1-12'); ap.add_argument('--dir', default='data/splits_r2/wf')
    a = ap.parse_args(); ov = json.loads(a.params)
    folds = load_folds(Path(a.dir), parse_range(a.folds))
    with ProcessPoolExecutor(12) as ex:
        res = list(ex.map(run, [a.strategy] * len(folds), [ov] * len(folds), folds))
    for (f, pnl, stt), fo in zip(res, folds):
        print(f"{fo['name']} pnl {pnl:+9.0f} fills {len(f):6d} lots {f.q.sum():7d}  {stt}")
    f = pd.concat([x[0] for x in res])
    f['sjb'] = pd.cut(f.sj, [-1, 1, 10, 60, 1e9], labels=['<1s', '1-10s', '10-60s', '>60s'])
    cols = [f'm{h:g}' for h in HS]
    g = f.groupby(['take', 'turb', 'sjb'], observed=True)
    out = g[cols].sum().round(0)
    out.insert(0, 'lots', g.q.sum())
    out['per_lot_1s'] = (out['m1'] / out['lots']).round(2)
    pd.set_option('display.width', 200)
    print(out.to_string())
    print('TOTAL', f[cols].sum().round(0).to_dict(), 'lots', int(f.q.sum()))
    by = f.groupby('fold')[cols].sum().round(0); print(by.to_string())


if __name__ == '__main__':
    main()
