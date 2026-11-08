"""Full-tape evaluation for round-3 strategies: PnL split by calm / volatile / shock window, per shock, per 500 s window.

    uv run python strategies/peter/r3_eval.py strategies/peter/r3_man.py '{}' '{"eta":0.1}' [--tapes real,synth] [--detail]

Tapes: "real" = data/round3_public.csv (labels from marek/research/round3/flow/*.csv), "synth" = data/synth/r3_v1_seed1-6
(labels from their .truth.json), or explicit csv paths. Shock window = onset - 1 s .. onset + 20 s. PnL is net of the breach
penalty. Several strategies: pass `path.py::{json}` items.
"""
from __future__ import annotations
import argparse, bisect, csv, json, statistics, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
R3 = ROOT / 'marek/research/round3/flow'


def ctx(tape: str):
    tj = (ROOT / tape).with_suffix('.truth.json')
    if tj.exists():
        tr = json.loads(tj.read_text())
        return sorted(tr['onsets']), [(a, b, 'volatile' if v else 'calm') for a, b, v in tr['segs']]
    on = [float(r['onset']) for r in csv.DictReader(open(R3 / 'shock_events.csv'))]
    reg = [(float(r['start']), float(r['end']), r['regime']) for r in csv.DictReader(open(R3 / 'regimes_1s.csv'))]
    return on, reg


def run(strategy: str, over: dict, tape: str) -> dict:
    from arena import load_agent
    from simulator.main import MAX_POSITION, MAX_POTENTIAL_POSITION, ORDER_ACCEPT_DELAY_MS, POSITION_BREACH_PENALTY_PER_SECOND
    from simulator.simulator import ExchangeSimulator, SimulationConfig
    strat = load_agent(Path(strategy), over, 'r3eval')
    cfg = SimulationConfig(max_position=MAX_POSITION, max_potential_position=MAX_POTENTIAL_POSITION,
                           position_breach_penalty_per_second=POSITION_BREACH_PENALTY_PER_SECOND,
                           order_accept_delay_ms=ORDER_ACCEPT_DELAY_MS)
    sim = ExchangeSimulator(strategy=strat, config=cfg)
    r = sim.run(ROOT / tape)
    ts = [x / 1e6 for x in sim._pnl_sample_times_us] + [sim.current_ts_us / 1e6]
    ps = list(sim._pnl_samples) + [r.pnl]
    on, reg = ctx(tape)
    starts = [a for a, b, g in reg]
    parts = {'calm': 0.0, 'volatile': 0.0, 'shock': 0.0}
    per_shock = [0.0] * len(on)
    for k in range(1, len(ts)):
        t = ts[k - 1]; d = ps[k] - ps[k - 1]
        i = bisect.bisect_right(on, t + 1.0) - 1
        if i >= 0 and on[i] - 1.0 <= t < on[i] + 20.0:
            parts['shock'] += d; per_shock[i] += d; continue
        j = bisect.bisect_right(starts, t) - 1
        parts[reg[j][2] if j >= 0 and t < reg[j][1] else 'calm'] += d
    win = []
    for w in range(12):
        a = bisect.bisect_right(ts, 500 * w) - 1; b = bisect.bisect_right(ts, 500 * (w + 1)) - 1
        win.append(ps[max(b, 0)] - ps[max(a, 0)])
    return {'tape': Path(tape).stem, 'pnl': r.pnl, 'breach': r.breach_penalty, 'fills': r.fills, 'parts': parts,
            'per_shock': per_shock, 'win': win, 'onsets': on}


def tapes_of(spec: str) -> list[str]:
    out = []
    for s in spec.split(','):
        if s == 'real':
            out.append('data/round3_public.csv')
        elif s == 'synth':
            out += [f'data/synth/r3_v1_seed{i}.csv' for i in range(1, 7)]
        else:
            out.append(s)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('strategy'); ap.add_argument('variants', nargs='*')
    ap.add_argument('--tapes', default='real'); ap.add_argument('--jobs', type=int, default=12)
    ap.add_argument('--detail', action='store_true'); ap.add_argument('--shocks', action='store_true')
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    specs = []
    for v in a.variants or ['{}']:
        if '::' in v:
            s, j = v.split('::', 1); specs.append((s, json.loads(j or '{}')))
        else:
            specs.append((a.strategy, json.loads(v)))
    tapes = tapes_of(a.tapes)
    base = None
    with ProcessPoolExecutor(a.jobs) as ex:
        futs = [[ex.submit(run, s, v, tp) for tp in tapes] for s, v in specs]
        for (s, v), fs in zip(specs, futs):
            res = [f.result() for f in fs]
            tot = {k: sum(r['parts'][k] for r in res) / len(res) for k in ('calm', 'volatile', 'shock')}
            pn = [r['pnl'] for r in res]
            line = (f"{Path(s).stem[:10]} {json.dumps(v)[:70]:70} mean {statistics.fmean(pn)/1e3:7.1f}k  "
                    f"calm {tot['calm']/1e3:6.1f} vol {tot['volatile']/1e3:6.1f} shock {tot['shock']/1e3:6.1f}  "
                    f"breach {statistics.fmean(r['breach'] for r in res)/1e3:5.1f}k")
            if base is None:
                base = pn
            else:
                d = [x - y for x, y in zip(pn, base)]
                line += f"  d {statistics.fmean(d)/1e3:+6.1f}k ({sum(x > 0 for x in d)}/{len(d)})"
            print(line, flush=True)
            if a.detail:
                print('    ' + ' '.join(f"{r['tape'][-6:]}={r['pnl']/1e3:.0f}k" for r in res))
            if a.shocks:
                for r in res:
                    print('    ' + r['tape'] + ' shocks: ' + ' '.join(f'{x/1e3:.1f}' for x in r['per_shock']))
            if a.out:
                with open(a.out, 'a') as fh:
                    fh.write(json.dumps({'strategy': s, 'params': v, 'res': res}) + '\n')


if __name__ == '__main__':
    main()
