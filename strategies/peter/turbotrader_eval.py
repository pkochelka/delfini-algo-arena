"""Paired evaluation of PARAMS variants against a base, with a result cache (turbotrader's selection harness).

    uv run python strategies/peter/turbotrader_eval.py strategies/peter/r3_turbotrader.py '{"k":1}' '{"k":2}' \
        [--base '{}'] [--tier sel|conf|r3|hold|r3full|quick|std|full] [--jobs 8]

Why fresh synthetic tapes: tuning on the one public round-3 tape overfits it (r3_grandpa: +26k on the public tape,
behind Adam's grandpa on the test set). The private tape is another draw of the same generator, so a choice should
be good in expectation over many draws. sel / conf are fresh draws of Marek's round-3 generator model
(marek/research/round3/strategy/r3_synth.py --duration 1500 --tag tt) that nothing was ever tuned on.
Caveat: a filter change that encodes that generator model wins on synth by construction (the ladder mixture: +53k
synth vs ~0 real), so it must also hold on the real tape.

Tiers:
    csel   (default) fresh tapes of the CALIBRATED generator strategies/peter/tt_synth.py, seeds 301-312, 1500 s each,
           trading from 30 s: SELECT on these. tt_synth = Marek's r3_synth with a ladder-length bug fixed, his final
           real-tape fit, and ladder chain / taker-tail laws matched to the real tape (tt_realsynth.py)
    cconf  calibrated generator, seeds 313-324: CONFIRM only, once per candidate
    sel    fresh draws of Marek's uncalibrated r3_synth, seeds 201-212 (it rates the ladder mixture +196k: its ladder
           bug turns most chains into singles, which is exactly what the mixture handles)
    conf   uncalibrated r3_synth seeds 213-224
    r3     real round-3 public wf01-12 (60 s warmup + ~370 s each): the "does not lose on the real tape" check
    hold   real round-3 holdout, 4 windows trading 4515-6000 s
    r3full the whole real round-3 tape in one run, PnL per 500 s piece (12 pieces)
    quick / std / full: the older r3_v1 synth windows mixed with r3 (see git history of grandpa_eval.py)
Every result is also split into the PnL earned within [onset - 1 s, onset + 20 s) of a shock (`sh`) and the rest
(`ex`): the shock sniper's PnL is chaotic, so filter or quoting changes are best judged on `ex`.
Profiling (2026-11-08): CSV parsing is ~3 % of a run, the simulator ~10 %, the strategy's own Python the rest.

Cache: logs/r3/tt_cache.json, keyed by sha1(strategy source) + params + window, so an unchanged base is never
re-run; saves merge with the file on disk, so concurrent runs do not drop each other's entries.
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
CACHE = ROOT / 'logs' / 'r3' / 'tt_cache.json'
FULL = ['data/round3_public.csv'] + [f'data/synth/r3_v1_seed{i}.csv' for i in range(1, 7)]
R3_ON = 'marek/research/round3/flow/shock_events.csv'
VER = 'v3'  # cache key version: v3 = shock split per part


def onsets(src: str) -> list[float]:
    if src == 'r3':
        return sorted(float(r['onset']) for r in csv.DictReader(open(ROOT / R3_ON)))
    return sorted(json.loads((ROOT / src).read_text())['onsets'])


def fresh(seeds, stem: str = 'r3_tt') -> list[dict]:
    return [{'name': f'{stem}{k}', 'path': f'data/synth/{stem}_seed{k}.csv', 'start': 30_000_000, 'group': 'synth',
             'on': f'data/synth/{stem}_seed{k}.truth.json'} for k in seeds]


def windows(tier: str) -> list[dict]:
    if tier == 'sel':
        return fresh(range(201, 213))
    if tier == 'conf':
        return fresh(range(213, 225))
    if tier == 'csel':  # calibrated generator (tt_synth.py): select
        return fresh(range(401, 413), 'tt_cal2')
    if tier == 'cconf':  # calibrated generator: confirm only
        return fresh(range(413, 425), 'tt_cal2')
    if tier == 'hold':  # round-3 holdout 4515-6000 s (never used for tuning): 4 windows
        d = ROOT / 'data/splits_r3/hold16'
        return [{'name': f'h_{f["name"]}', 'path': f'data/splits_r3/hold16/{f["file"]}', 'start': f['trade_start_us'],
                 'group': 'r3', 'on': 'r3'} for f in json.loads((d / 'folds.json').read_text())['folds']
                if (d / f['file']).exists()]
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
                            'start': f['trade_start_us'], 'group': 'synth',
                            'on': f'data/synth/r3_v1_seed{i}.truth.json'})
    return out


def run_one(strategy: str, over: dict, w: dict, win_s: float) -> dict:
    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(ROOT))
    from arena import load_agent
    from simulator.main import (MAX_POSITION, MAX_POTENTIAL_POSITION, ORDER_ACCEPT_DELAY_MS,
                                POSITION_BREACH_PENALTY_PER_SECOND)
    from simulator.simulator import ExchangeSimulator, SimulationConfig

    strat = load_agent(Path(strategy), over | {'trade_start_us': w['start']}, 'tt')
    cfg = SimulationConfig(max_position=MAX_POSITION, max_potential_position=MAX_POTENTIAL_POSITION,
                           position_breach_penalty_per_second=POSITION_BREACH_PENALTY_PER_SECOND,
                           order_accept_delay_ms=ORDER_ACCEPT_DELAY_MS)
    sim = ExchangeSimulator(strategy=strat, config=cfg)
    r = sim.run(ROOT / w['path'])
    curve = sim.pnl_curve(r.pnl)
    on = onsets(w['on'])
    t0 = curve[0][0]
    n = int((curve[-1][0] - t0) // (win_s * 1e6)) + 1 if win_s else 1  # whole tape: win_s pieces
    parts, shock = [0.0] * n, [0.0] * n
    prev_t, prev_v = None, 0.0
    for ts, v in curve:  # each 1 s PnL increment goes to its piece, and to `shock` if it starts in a shock window
        if prev_t is not None:
            k = min(int((prev_t - t0) // (win_s * 1e6)), n - 1) if win_s else 0
            parts[k] += v - prev_v
            s = prev_t * 1e-6
            i = bisect.bisect_right(on, s + 1.0) - 1
            if i >= 0 and on[i] - 1.0 <= s < on[i] + 20.0:
                shock[k] += v - prev_v
        prev_t, prev_v = ts, v
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
    ap.add_argument('--tier', default='csel',
                    choices=['csel', 'cconf', 'sel', 'conf', 'r3', 'hold', 'r3full', 'quick', 'std', 'full'])
    ap.add_argument('--jobs', type=int, default=8)
    ap.add_argument('--no-cache', action='store_true')
    a = ap.parse_args()
    if a.tier in ('full', 'r3full'):
        ws = [{'name': Path(p).stem, 'path': p, 'start': 0, 'group': 'r3' if 'round3' in p else 'synth',
               'on': 'r3' if 'round3' in p else p.replace('.csv', '.truth.json')}
              for p in (FULL if a.tier == 'full' else FULL[:1])]
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
        mixed = len({w['group'] for w in ws}) > 1
        print(f"[{a.tier}: {len(ws)} windows, {len(futs)} runs] base {json.dumps(base)[:80]}: total "
              f"{sum(r['pnl'] for r in b) / 1e3:.1f}k" + (f" (r3 {grp(b, 'r3') / 1e3:.1f}k, synth {grp(b, 'synth') / 1e3:.1f}k)" if mixed else '')
              + f"; shock part {sum(sum(r['shock']) for r in b) / 1e3:.1f}k, breach {sum(r['breach'] for r in b) / 1e3:.1f}k",
              flush=True)
        for v, ks in zip(vs[1:], keys[1:]):
            rs = [get(k) for k in ks]
            dw = [p - q for x, y in zip(rs, b) for p, q in zip(x['parts'], y['parts'])]
            dx = [(p - ps) - (q - qs) for x, y in zip(rs, b)
                  for p, ps, q, qs in zip(x['parts'], x['shock'], y['parts'], y['shock'])]
            dsh = sum(sum(x['shock']) - sum(y['shock']) for x, y in zip(rs, b))
            line = (f"{json.dumps(v)[:66]:66} d {sum(dw) / 1e3:+6.1f}k t {tstat(dw):+5.1f} won {sum(x > 0 for x in dw):2d}/{len(dw)}"
                    f" | ex {sum(dx) / 1e3:+6.1f}k t {tstat(dx):+5.1f} won {sum(x > 0 for x in dx):2d} | sh {dsh / 1e3:+6.1f}k")
            if mixed:
                line += f" | r3 {(grp(rs, 'r3') - grp(b, 'r3')) / 1e3:+6.1f}k synth {(grp(rs, 'synth') - grp(b, 'synth')) / 1e3:+6.1f}k"
            print(line + f" | breach {sum(r['breach'] for r in rs) / 1e3:.1f}k", flush=True)


if __name__ == '__main__':
    main()
