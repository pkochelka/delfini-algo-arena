"""Generate synthetic tapes from the round-1 generator model (fitted by r1_model.py).

Usage:
    uv run python strategies/peter/r1_synth.py --seed 1                                  # round1 parameters
    uv run python strategies/peter/r1_synth.py --seed 2 --set taker_rate=6 c=0.035 --duration 3000
    uv run python strategies/peter/r1_synth.py --params strategies/peter/tuned/round1_model.json --seeds 1 2 3

Writes data/synth/r1_<tag>_seed<k>.csv (tag from --tag, default 'base'), in the backtest format.
Every knob of the model is a parameter, so one can be moved at a time (see PARAMS).

The model:
    S          starts at s0; every taker moves it by c * sign * filled**impact_exp
               (+ optional Brownian s_sigma ticks/sqrt(s), applied between events)
    makers     Poisson(maker_rate), buy with prob 1/2, size max(1, round(lognormal(maker_mu, maker_sd))),
               bid = min(floor(S - D), best_ask - 1), ask = max(ceil(S + D), best_bid + 1),
               D = d_shift + Exp(mean d_scale); a cancel decision comes with hazard
               cancel_alpha + cancel_beta * max(0, adverse move of S since the insert), and the cancel row
               follows Exp(rate life_b) later; it is written even if the order was filled meanwhile (as in round 1)
    takers     Poisson(taker_rate) per side, size max(1, round(lognormal(taker_mu, taker_sd))),
               IOC at round(S + taker_off) (buy) / round(S - taker_off) (sell), matched FIFO
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
from bisect import insort
from collections import deque
from pathlib import Path

import numpy as np

PARAMS: dict[str, float] = {
    'duration': 12000.0,
    's0': 10000.0,
    'c': 0.026,              # impact per sqrt(lot) filled, ticks
    'impact_exp': 0.5,
    's_sigma': 0.0,          # extra Brownian motion of S, ticks / sqrt(s) (0 in round 1)
    'maker_rate': 180.0,     # inserts / s, both sides together
    'maker_mu': 2.6581, 'maker_sd': 1.7991,
    'd_shift': 0.05, 'd_scale': 0.45,
    # cancel: a decision arrives with hazard cancel_alpha + cancel_beta * max(0, adverse move of S since insert),
    # then the cancel is sent after Exp(rate life_b). cancel_beta = 0 gives a state-free Exp(alpha) + Exp(b) life.
    # calibrated by matching round-1 statistics (see round1_generator.md): alpha 1.72-1.76, beta 5.8-6, b 35-40
    'cancel_alpha': 1.72, 'cancel_beta': 5.8, 'life_b': 40.0,
    'taker_rate': 4.0,       # IOCs / s per side
    'taker_mu': 4.1058, 'taker_sd': 1.3959,
    'taker_off': 1.0,
    'first_id': 1000001,
}


def load_params(path: Path | None) -> dict[str, float]:
    p = dict(PARAMS)
    if path is None:
        return p
    m = json.loads(path.read_text())
    p.update({
        's0': m['S']['s0'], 'c': m['S']['c'],
        'maker_rate': m['arrivals']['maker_rate'], 'taker_rate': m['arrivals']['taker_rate_per_side'],
        'maker_mu': m['maker_size']['mu'], 'maker_sd': m['maker_size']['sd'],
        'taker_mu': m['taker_size']['mu'], 'taker_sd': m['taker_size']['sd'],
        'd_shift': m['maker_offset']['d_shift'], 'd_scale': m['maker_offset']['d_scale'],
        'duration': round(m['duration_s']),
    })
    if 'cancel' in m:
        p.update({k: m['cancel'][k] for k in ('cancel_alpha', 'cancel_beta', 'life_b') if k in m['cancel']})
    return p


def generate(p: dict[str, float], seed: int) -> list[str]:
    rng = np.random.default_rng(seed)
    T = p['duration']
    # arrival streams: 0 maker, 1 taker buy, 2 taker sell
    times, kinds = [], []
    for kind, rate in ((0, p['maker_rate']), (1, p['taker_rate']), (2, p['taker_rate'])):
        n = rng.poisson(rate * T)
        times.append(np.sort(rng.uniform(0, T, n))); kinds.append(np.full(n, kind, dtype=np.int8))
    t_all = np.concatenate(times); k_all = np.concatenate(kinds)
    order = np.argsort(t_all, kind='stable'); t_all = t_all[order].tolist(); k_all = k_all[order].tolist()
    nm = sum(1 for k in k_all if k == 0); nt = len(k_all) - nm
    # pre-drawn randoms
    m_side = (rng.random(nm) < 0.5).tolist()
    m_vol = np.maximum(1, np.rint(np.exp(rng.normal(p['maker_mu'], p['maker_sd'], nm)))).astype(np.int64).tolist()
    m_D = (p['d_shift'] + rng.exponential(p['d_scale'], nm)).tolist()
    m_E = rng.exponential(1.0, nm).tolist()                      # integrated-hazard budget of the cancel decision
    m_delay = rng.exponential(1 / p['life_b'], nm).tolist()      # decision -> cancel row
    t_vol = np.maximum(1, np.rint(np.exp(rng.normal(p['taker_mu'], p['taker_sd'], nt)))).astype(np.int64).tolist()
    s_noise = rng.normal(0, 1, len(t_all)).tolist() if p['s_sigma'] > 0 else None

    S = p['s0']; c = p['c']; ex = p['impact_exp']; off = p['taker_off']; sig = p['s_sigma']
    levels = ({}, {})            # side -> price -> deque of [oid, remaining]
    qty = ({}, {})
    prices = ([], [])            # ascending
    live: dict[int, list] = {}
    cancels: list[tuple[float, int]] = []
    alpha, beta = p['cancel_alpha'], p['cancel_beta']
    pending: dict[int, list] = {}            # oid -> [sign (+1 bid), S at insert, hazard, t of last update, budget left, version, delay]
    decisions: list[tuple[float, int, int]] = []   # (decision time, oid, version)
    rows: list[str] = []
    oid = int(p['first_id']); mi = ti = 0
    last_us = -1
    t_prev = 0.0

    def stamp(t: float) -> str:
        nonlocal last_us
        us = int(round(t * 1e6))
        if us <= last_us:
            us = last_us + 1
        last_us = us
        return f'{us // 1_000_000}.{us % 1_000_000:06d}'

    def do_cancel(t: float, o: int) -> None:
        rows.append(f'{stamp(t)},Cancel,{o},,,,')
        e = live.pop(o, None)
        if e is None:
            return
        s, px, entry = e
        if entry[1] > 0:
            qty[s][px] -= entry[1]
            entry[1] = 0
            if qty[s][px] == 0:
                del levels[s][px]; del qty[s][px]; prices[s].remove(px)

    def flush(t: float) -> None:
        """Process cancel decisions and cancel rows due before t, in time order."""
        while True:
            td = decisions[0][0] if decisions else math.inf
            tc = cancels[0][0] if cancels else math.inf
            if min(td, tc) >= t:
                return
            if td <= tc:
                _, o, ver = heapq.heappop(decisions)
                st = pending.get(o)
                if st is not None and st[5] == ver:
                    del pending[o]
                    heapq.heappush(cancels, (td + st[6], o))
            else:
                _, o = heapq.heappop(cancels)
                do_cancel(tc, o)

    def on_jump(t: float) -> None:
        """S moved: re-rate the pending cancel decisions (piecewise-constant hazard, memoryless)."""
        for o, st in pending.items():
            mv = (st[1] - S) if st[0] > 0 else (S - st[1])
            lam = alpha + beta * mv if mv > 0 else alpha
            if lam != st[2]:
                st[4] -= st[2] * (t - st[3])
                st[2] = lam; st[3] = t; st[5] += 1
                heapq.heappush(decisions, (t + max(st[4], 0.0) / lam, o, st[5]))

    for i, t in enumerate(t_all):
        flush(t)
        if sig > 0:
            S += sig * math.sqrt(t - t_prev) * s_noise[i]
        t_prev = t
        kind = k_all[i]
        if kind == 0:
            buy = m_side[mi]; v = m_vol[mi]; D = m_D[mi]; E = m_E[mi]; delay = m_delay[mi]; mi += 1
            if buy:
                px = math.floor(S - D)
                if prices[1]:
                    px = min(px, prices[1][0] - 1)
                s = 0
            else:
                px = math.ceil(S + D)
                if prices[0]:
                    px = max(px, prices[0][-1] + 1)
                s = 1
            if px <= 0:
                continue
            rows.append(f'{stamp(t)},Insert,{oid},{"B" if buy else "A"},{v},{px},G')
            lv = levels[s]
            if px not in lv:
                lv[px] = deque(); qty[s][px] = 0; insort(prices[s], px)
            entry = [oid, v]
            lv[px].append(entry); qty[s][px] += v
            live[oid] = (s, px, entry)
            pending[oid] = [1 if buy else -1, S, alpha, t, E, 0, delay]
            heapq.heappush(decisions, (t + E / alpha, oid, 0))
            oid += 1
        else:
            buy = kind == 1; v = t_vol[ti]; ti += 1
            px = round(S + off) if buy else round(S - off)
            rows.append(f'{stamp(t)},Insert,{oid},{"B" if buy else "A"},{v},{px},F')
            oid += 1
            opp = 1 if buy else 0
            op = prices[opp]; filled = 0
            while v > 0 and op:
                best = op[0] if buy else op[-1]
                if (best > px) if buy else (best < px):
                    break
                q = levels[opp][best]
                while v > 0 and q:
                    m = q[0]
                    if m[1] == 0:
                        q.popleft(); continue
                    x = v if v < m[1] else m[1]
                    v -= x; m[1] -= x; qty[opp][best] -= x; filled += x
                    if m[1] == 0:
                        q.popleft(); live.pop(m[0], None)
                if qty[opp][best] == 0:
                    del levels[opp][best]; del qty[opp][best]
                    if buy:
                        op.pop(0)
                    else:
                        op.pop()
            if filled:
                S += (c if buy else -c) * filled ** ex
                if beta:
                    on_jump(t)
    flush(T)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--params', type=Path, default=None, help='JSON from r1_model.py (default: built-in round-1 values)')
    ap.add_argument('--set', nargs='*', default=[], metavar='KEY=VALUE', help=f'override: {", ".join(PARAMS)}')
    ap.add_argument('--duration', type=float, default=None)
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--seeds', type=int, nargs='*', default=None)
    ap.add_argument('--tag', default='base')
    ap.add_argument('--out-dir', type=Path, default=Path('data/synth'))
    args = ap.parse_args()
    p = load_params(args.params)
    for kv in args.set:
        k, v = kv.split('=')
        if k not in PARAMS:
            raise SystemExit(f'unknown parameter {k}')
        p[k] = float(v)
    if args.duration:
        p['duration'] = args.duration
    seeds = args.seeds or [args.seed if args.seed is not None else 0]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        rows = generate(p, seed)
        out = args.out_dir / f'r1_{args.tag}_seed{seed}.csv'
        with out.open('w', newline='') as fh:
            fh.write('Time,Operation,OrderId,Side,Volume,Price,Lifespan\n')
            fh.write('\n'.join(rows)); fh.write('\n')
        (out.with_suffix('.params.json')).write_text(json.dumps({**p, 'seed': seed}, indent=2))
        print(f'{out}: {len(rows):,} rows')


if __name__ == '__main__':
    main()
