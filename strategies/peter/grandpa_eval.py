"""Fast paired evaluation of PARAMS variants against a base, on short round-3 windows, with a result cache.

    uv run python strategies/peter/grandpa_eval.py strategies/peter/r3_grandpa.py '{"k":1}' '{"k":2}' \
        [--base '{}'] [--tier r3|quick|std|full] [--jobs 8]

Tiers (every window = 60 s warmup + ~370 s traded; the strategy only trades from trade_start_us):
    r3     round-3 public wf01-12 (default). The synth tapes come from our own generator model, so a filter change
           that encodes the same model scores well there by construction; the real tape is the arbiter.
    quick  round-3 public wf01,03,..,11 (6) + 2 windows per synthetic tape (12)        ~ 7.7k s of tape
    std    round-3 public wf01-12 (12)  + 4 windows per synthetic tape (24)            ~ 15.5k s
    full   the 7 whole tapes (round 3 public + synth seeds 1-6), PnL per 500 s window  ~ 42k s
The windows come from split_data.py (data/splits_r3/wf, data/splits_r3/synth/s<k>, staggered so the seeds cover
different parts of their tapes). Profiling (2026-11-08): CSV parsing is ~3 % of a run and the simulator ~10 %; the
strategy's per-event Python is the rest, so the only real speed-ups are less tape and not re-running the base.

Shock split: every window's PnL is also split into the part earned within [onset - 1 s, onset + 20 s) of a shock
(onsets: marek/research/round3/flow/shock_events.csv, synth .truth.json) and the rest. The shock sniper's PnL is
chaotic (it swings by ~20k between near-identical configs), so a filter or quoting change is best judged on the
non-shock part: `ex` = its difference to the base, with its own t over the windows; `sh` = the shock part.

Cache: logs/r3/gp_cache.json, keyed by sha1(strategy source) + params + window, so an unchanged base (or any
repeated variant) is never re-run. Editing the strategy file changes the hash and invalidates its entries.
Saves merge with the file on disk, so concurrent runs do not drop each other's entries.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
import os
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
CACHE = ROOT / 'logs' / 'r3' / 'gp_cache.json'
FULL = ['data/round3_public.csv'] + [f'data/synth/r3_v1_seed{i}.csv' for i in range(1, 7)]
VER = 'v2'  # cache key version: v2 adds the shock split


def onsets(group: str) -> list[float]:
    if group == 'r3':
        f = ROOT / 'marek/research/round3/flow/shock_events.csv'
        return sorted(float(r['onset']) for r in csv.DictReader(open(f)))
    return sorted(json.loads((ROOT / f'data/synth/r3_v1_seed{group[1:]}.truth.json').read_text())['onsets'])


def windows(tier: str) -> list[dict]:
    out = []
    r3 = json.loads((ROOT / 'data/splits_r3/wf/folds.json').read_text())['folds']
    for k, f in enumerate(r3):
        if tier in ('std', 'r3') or k % 2 == 0:
            out.append({'name': f'r3_{f["name"]}', 'path': f'data/splits_r3/wf/{f["file"]}',
                        'start': f['trade_start_us'], 'group': 'r3', 'on': 'r3'})
    if tier == 'r3':
        return out
    for i in range(1, 7):
        d = ROOT / f'data/splits_r3/synth/s{i}'
        fs = [f for f in json.loads((d / 'folds.json').read_text())['folds'] if (d / f['file']).exists()]
        for k, f in enumerate(fs):
            if tier == 'std' or k % 2 == 0:
                out.append({'name': f's{i}_{f["name"]}', 'path': f'data/splits_r3/synth/s{i}/{f["file"]}',
                            'start': f['trade_start_us'], 'group': 'synth', 'on': f's{i}'})
    return out


def run_one(strategy: str, over: dict, w: dict, win_s: float) -> dict:
    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(ROOT))
    from arena import load_agent
    from simulator.main import (MAX_POSITION, MAX_POTENTIAL_POSITION, ORDER_ACCEPT_DELAY_MS,
                                POSITION_BREACH_PENALTY_PER_SECOND)
    from simulator.simulator import ExchangeSimulator, SimulationConfig

    strat = load_agent(Path(strategy), over | {'trade_start_us': w['start']}, 'gp')
    cfg = SimulationConfig(max_position=MAX_POSITION, max_potential_position=MAX_POTENTIAL_POSITION,
                           position_breach_penalty_per_second=POSITION_BREACH_PENALTY_PER_SECOND,
                           order_accept_delay_ms=ORDER_ACCEPT_DELAY_MS)
    sim = ExchangeSimulator(strategy=strat, config=cfg)
    r = sim.run(ROOT / w['path'])
    curve = sim.pnl_curve(r.pnl)
    on = onsets(w['on'])
    shock, prev_t, prev_v = 0.0, None, 0.0
    for ts, v in curve:  # PnL increments whose interval starts inside a shock window
        if prev_t is not None:
            s = prev_t * 1e-6
            i = bisect.bisect_right(on, s + 1.0) - 1
            if i >= 0 and on[i] - 1.0 <= s < on[i] + 20.0:
                shock += v - prev_v
        prev_t, prev_v = ts, v
    parts = [r.pnl]
    if win_s:  # whole tape: split the PnL curve into win_s windows
        t0, parts, k, last = curve[0][0], [], 1, 0.0
        for ts, v in curve:
            while ts - t0 >= k * win_s * 1e6:
                parts.append(v - last)
                last, k = v, k + 1
        parts.append(r.pnl - last)
    return {'pnl': r.pnl, 'parts': parts, 'shock': shock, 'breach': r.breach_penalty, 'fills': r.fills}


def save(new: dict) -> None:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    disk = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    disk.update(new)
    tmp = CACHE.with_suffix(f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(disk))
    os.replace(tmp, CACHE)


def tstat(d: list[float]) -> float:
    sd = statistics.stdev(d) if len(d) > 1 else 0.0
    return statistics.fmean(d) / (sd / math.sqrt(len(d))) if sd else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('strategy')
    ap.add_argument('variants', nargs='*')
    ap.add_argument('--base', default='{}')
    ap.add_argument('--tier', default='r3', choices=['r3', 'quick', 'std', 'full'])
    ap.add_argument('--jobs', type=int, default=8)
    ap.add_argument('--no-cache', action='store_true')
    a = ap.parse_args()
    if a.tier == 'full':
        ws = [{'name': Path(p).stem, 'path': p, 'start': 0, 'group': 'r3' if 'round3' in p else 'synth',
               'on': 'r3' if 'round3' in p else f's{p[-5]}'} for p in FULL]
        win_s = 500.0
    else:
        ws, win_s = windows(a.tier), 0.0
    src = hashlib.sha1(Path(a.strategy).read_bytes()).hexdigest()[:12]
    cache = {} if a.no_cache or not CACHE.exists() else json.loads(CACHE.read_text())
    base = json.loads(a.base)
    vs = [{}] + [json.loads(v) for v in a.variants]
    keys = [[f"{src}|{json.dumps(base | v, sort_keys=True)}|{w['path']}|{w['start']}|{win_s}|{VER}" for w in ws]
            for v in vs]
    with ProcessPoolExecutor(a.jobs) as ex:
        futs = {}
        for v, ks in zip(vs, keys):
            for w, k in zip(ws, ks):
                if k not in cache and k not in futs:
                    futs[k] = ex.submit(run_one, a.strategy, base | v, w, win_s)

        def get(k):
            if k not in cache:
                cache[k] = futs[k].result()
                save({k: cache[k]})
            return cache[k]

        b = [get(k) for k in keys[0]]
        grp = lambda rs, g: sum(r['pnl'] for r, w in zip(rs, ws) if w['group'] == g)  # noqa: E731
        print(f"[{a.tier}: {len(ws)} windows, {len(futs)} runs] base {json.dumps(base)[:80]}: total "
              f"{sum(r['pnl'] for r in b) / 1e3:.1f}k (r3 {grp(b, 'r3') / 1e3:.1f}k, synth {grp(b, 'synth') / 1e3:.1f}k;"
              f" shock part {sum(r['shock'] for r in b) / 1e3:.1f}k), breach {sum(r['breach'] for r in b) / 1e3:.1f}k",
              flush=True)
        for v, ks in zip(vs[1:], keys[1:]):
            rs = [get(k) for k in ks]
            dw = [p - q for x, y in zip(rs, b) for p, q in zip(x['parts'], y['parts'])]
            dx = [(x['pnl'] - x['shock']) - (y['pnl'] - y['shock']) for x, y in zip(rs, b)]
            dsh = sum(x['shock'] - y['shock'] for x, y in zip(rs, b))
            line = (f"{json.dumps(v)[:66]:66} d {sum(dw) / 1e3:+6.1f}k t {tstat(dw):+5.1f} won {sum(x > 0 for x in dw):2d}/{len(dw)}"
                    f" | ex {sum(dx) / 1e3:+6.1f}k t {tstat(dx):+5.1f} won {sum(x > 0 for x in dx):2d} | sh {dsh / 1e3:+6.1f}k")
            if a.tier != 'r3':
                line += f" | r3 {(grp(rs, 'r3') - grp(b, 'r3')) / 1e3:+6.1f}k synth {(grp(rs, 'synth') - grp(b, 'synth')) / 1e3:+6.1f}k"
            print(line + f" | breach {sum(r['breach'] for r in rs) / 1e3:.1f}k", flush=True)


if __name__ == '__main__':
    main()
