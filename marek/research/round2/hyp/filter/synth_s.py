"""Synthetic round-2 tapes: the round-1 generator (strategies/peter/r1_synth.py) + calm/volatile regime + shocks.

Usage:
    uv run python marek/research/round2/r2_synth.py --seeds 1 2 3 [--duration 6000] [--set vol_c=0.15 ...]

Writes data/synth/r2_<tag>_seed<k>.csv (+ .params.json and .truth.json with regime segments and shock onsets).

Model: the final fit in marek/research/round2/round2_generator.md and s_dynamics/params_v3.json.
    regime     2-state Markov chain, calm -> vol p_cv /s, vol -> calm p_vc /s (frozen during shock windows)
    impact     each FILLED taker moves S by sign * c * filled**e + N(0, xi); unfilled IOCs move nothing
    per regime c, e, xi, D = d_shift + Exp(d_scale) (maker offset), taker_rate per side, cancel alpha / beta,
               taker limit round(S +- off), and with prob eps a further ceil(0.5 + Exp(tau)) ticks (aggressive tail);
               volatile also has small Laplace(scale bj) jumps of S at rate lamj /s (fit, possibly a misfit);
               optional Brownian diffusion sig (ticks/sqrt(s), 0 in the final fit)
    shocks     first onset U(gap_lo, gap_hi), then gaps U(gap_lo, gap_hi). Drought: maker rate x drought_mult and
               cancel alpha = drought_alpha, D = drought_d_shift + Exp(drought_d_scale) for drought_s seconds; then
               S jumps by N(0, jump_sd) and a symmetric taker burst follows (burst_rate total for burst_flat_s,
               falling linearly to 0 at burst_end_s).
    cancel     hazard alpha(state now) + beta(state now) * max(0, adverse S move since insert); row after
               dead + Exp(life_b) (written even if the order filled).
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
    'duration': 6000.0,
    's0': 10000.0,
    'maker_rate': 180.0,
    'maker_mu': 2.657, 'maker_sd': 1.803,
    'taker_mu': 4.10, 'taker_sd': 1.40,
    'life_b': 50.0, 'dead': 0.005,
    # calm
    'calm_c': 0.0142, 'calm_e': 0.625, 'calm_xi': 0.124, 'calm_sig': 0.0,
    'calm_d_shift': 0.05, 'calm_d_scale': 0.45, 'calm_off': 1.0, 'calm_eps': 0.022, 'calm_tau': 3.15,
    'calm_taker_rate': 4.0, 'calm_alpha': 1.73, 'calm_beta': 6.1, 'calm_lamj': 0.0, 'calm_bj': 1.0,
    # volatile
    'vol_c': 0.0741, 'vol_e': 0.61, 'vol_xi': 0.28, 'vol_sig': 0.0,
    'vol_d_shift': 0.70, 'vol_d_scale': 0.90, 'vol_off': 2.55, 'vol_eps': 0.04, 'vol_tau': 6.27,
    'vol_taker_rate': 6.0, 'vol_alpha': 1.9, 'vol_beta': 4.1, 'vol_lamj': 0.23, 'vol_bj': 0.84,
    'p_cv': 0.0040, 'p_vc': 0.023,
    # shocks
    'shocks': 1.0,
    'gap_lo': 150.0, 'gap_hi': 300.0,
    'drought_s': 3.0, 'drought_mult': 0.30, 'drought_alpha': 6.3,
    'drought_d_shift': 0.54, 'drought_d_scale': 1.33,
    'jump_sd': 43.0,
    'burst_rate': 35.0, 'burst_flat_s': 1.25, 'burst_end_s': 3.0,
    'first_id': 1000001,
}


def schedule(p, rng):
    """Shock onsets and regime segments [(start, end, vol)]."""
    T = p['duration']
    onsets = []
    if p['shocks']:
        t = rng.uniform(p['gap_lo'], p['gap_hi'])
        while t < T - 10:
            onsets.append(t)
            t += rng.uniform(p['gap_lo'], p['gap_hi'])
    segs = []
    t, vol, start = 0.0, False, 0.0
    while t < T:
        t += 1.0
        if any(o <= t < o + p['drought_s'] + p['burst_end_s'] for o in onsets):
            continue
        if rng.random() < (p['p_vc'] if vol else p['p_cv']):
            segs.append((start, t, vol))
            start, vol = t, not vol
    segs.append((start, T, vol))
    return onsets, segs


def generate(p: dict[str, float], seed: int):
    rng = np.random.default_rng(seed)
    T = p['duration']
    onsets, segs = schedule(p, rng)
    jumps = rng.normal(0, p['jump_sd'], len(onsets)).tolist()

    # state-change times: ('reg', t, vol) and ('drought', t, on) and ('jump', t, J)
    changes = []
    for a, b, v in segs[1:]:
        changes.append((a, 0, v))
    for o, J in zip(onsets, jumps):
        changes.append((o, 1, 1))
        changes.append((o + p['drought_s'], 1, 0))
        changes.append((o + p['drought_s'], 2, J))
    changes.sort()

    # candidate arrivals at max rates, thinned below
    mr, tr_max = p['maker_rate'], max(p['calm_taker_rate'], p['vol_taker_rate']) + p['burst_rate'] / 2
    lj_max = max(p['calm_lamj'], p['vol_lamj'])
    times, kinds = [], []
    for kind, rate in ((0, mr), (1, tr_max), (2, tr_max), (3, lj_max)):
        n = rng.poisson(rate * T)
        times.append(np.sort(rng.uniform(0, T, n)))
        kinds.append(np.full(n, kind, dtype=np.int8))
    t_all = np.concatenate(times); k_all = np.concatenate(kinds)
    order = np.argsort(t_all, kind='stable'); t_all = t_all[order].tolist(); k_all = k_all[order].tolist()
    u_thin = rng.random(len(t_all)).tolist()
    n = len(t_all)
    m_side = (rng.random(n) < 0.5).tolist()
    m_vol = np.maximum(1, np.rint(np.exp(rng.normal(p['maker_mu'], p['maker_sd'], n)))).astype(np.int64).tolist()
    m_E = rng.exponential(1.0, n).tolist()
    m_Dx = rng.exponential(1.0, n).tolist()
    m_delay = (p['dead'] + rng.exponential(1 / p['life_b'], n)).tolist()
    t_vol = np.maximum(1, np.rint(np.exp(rng.normal(p['taker_mu'], p['taker_sd'], n)))).astype(np.int64).tolist()
    s_noise = rng.normal(0, 1, n).tolist()
    xi_noise = rng.normal(0, 1, n).tolist()
    t_tail = rng.random(n).tolist()
    t_extra = rng.exponential(1.0, n).tolist()
    j_lap = rng.laplace(0, 1, n).tolist()

    st = {'vol': segs[0][2], 'drought': False, 'burst_t0': -1e9}
    S = p['s0']
    SP = [(0.0, S)]
    levels = ({}, {}); qty = ({}, {}); prices = ([], [])
    live: dict[int, tuple] = {}
    cancels: list = []
    pending: dict[int, list] = {}   # oid -> [sign, S at insert, lam, t last, budget, version, delay]
    decisions: list = []
    rows: list[str] = []
    oid = int(p['first_id'])
    last_us = -1
    t_prev = 0.0
    ci = 0

    def pref():
        return 'vol_' if st['vol'] else 'calm_'

    def ab():
        if st['drought']:
            return p['drought_alpha'], p['vol_beta']
        return p[pref() + 'alpha'], p[pref() + 'beta']

    def stamp(t):
        nonlocal last_us
        us = int(round(t * 1e6))
        if us <= last_us:
            us = last_us + 1
        last_us = us
        return f'{us // 1_000_000}.{us % 1_000_000:06d}'

    def do_cancel(t, o):
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

    def flush(t):
        while True:
            td = decisions[0][0] if decisions else math.inf
            tc = cancels[0][0] if cancels else math.inf
            if min(td, tc) >= t:
                return
            if td <= tc:
                _, o, ver = heapq.heappop(decisions)
                q = pending.get(o)
                if q is not None and q[5] == ver:
                    del pending[o]
                    heapq.heappush(cancels, (td + q[6], o))
            else:
                _, o = heapq.heappop(cancels)
                do_cancel(tc, o)

    def rerate(t):
        alpha, beta = ab()
        for o, q in pending.items():
            mv = (q[1] - S) if q[0] > 0 else (S - q[1])
            lam = alpha + beta * mv if mv > 0 else alpha
            if lam != q[2]:
                q[4] -= q[2] * (t - q[3])
                q[2] = lam; q[3] = t; q[5] += 1
                heapq.heappush(decisions, (t + max(q[4], 0.0) / lam, o, q[5]))

    for i, t in enumerate(t_all):
        # state changes before t
        while ci < len(changes) and changes[ci][0] <= t:
            tc, kind, val = changes[ci]; ci += 1
            flush(tc)
            if kind == 0:
                st['vol'] = bool(val)
            elif kind == 1:
                st['drought'] = bool(val)
            else:
                S += val
                SP.append((tc, S))
                st['burst_t0'] = tc
            rerate(tc)
        flush(t)
        sig = p[pref() + 'sig']
        if sig > 0 and t > t_prev:
            S += sig * math.sqrt(t - t_prev) * s_noise[i]
            SP.append((t, S))
            if i % 50 == 0:
                rerate(t)
        t_prev = t
        kind = k_all[i]
        if kind == 3:
            lam = p[pref() + 'lamj']
            if st['drought'] or u_thin[i] > lam / lj_max:
                continue
            S += p[pref() + 'bj'] * j_lap[i]
            SP.append((t, S))
            rerate(t)
            continue
        if kind == 0:
            if st['drought'] and u_thin[i] > p['drought_mult']:
                continue
            buy = m_side[i]; v = m_vol[i]
            if st['drought']:
                D = p['drought_d_shift'] + p['drought_d_scale'] * m_Dx[i]
            else:
                D = p[pref() + 'd_shift'] + p[pref() + 'd_scale'] * m_Dx[i]
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
            alpha, _ = ab()
            pending[oid] = [1 if buy else -1, S, alpha, t, m_E[i], 0, m_delay[i]]
            heapq.heappush(decisions, (t + m_E[i] / alpha, oid, 0))
            oid += 1
        else:
            rate = p[pref() + 'taker_rate']
            dtb = t - st['burst_t0']
            if 0 <= dtb < p['burst_end_s']:
                f = 1.0 if dtb < p['burst_flat_s'] else (p['burst_end_s'] - dtb) / (p['burst_end_s'] - p['burst_flat_s'])
                rate += p['burst_rate'] / 2 * f
            if u_thin[i] > rate / tr_max:
                continue
            buy = kind == 1; v = t_vol[i]
            off = p[pref() + 'off']
            px = round(S + off) if buy else round(S - off)
            if t_tail[i] < p[pref() + 'eps']:
                x = math.ceil(0.5 + p[pref() + 'tau'] * t_extra[i])
                px += x if buy else -x
            if px <= 0:
                continue
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
                c = p[pref() + 'c']
                S += (c if buy else -c) * filled ** p[pref() + 'e'] + p[pref() + 'xi'] * xi_noise[i]
                SP.append((t, S))
                rerate(t)
    flush(T)
    truth = {'onsets': onsets, 'jumps': jumps, 'segs': segs}
    truth['S'] = SP
    return rows, truth


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--set', nargs='*', default=[], metavar='KEY=VALUE')
    ap.add_argument('--duration', type=float, default=None)
    ap.add_argument('--seeds', type=int, nargs='*', default=[0])
    ap.add_argument('--tag', default='base')
    ap.add_argument('--out-dir', type=Path, default=Path('data/synth'))
    args = ap.parse_args()
    p = dict(PARAMS)
    for kv in args.set:
        k, v = kv.split('=')
        if k not in PARAMS:
            raise SystemExit(f'unknown parameter {k}')
        p[k] = float(v)
    if args.duration:
        p['duration'] = args.duration
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        rows, truth = generate(p, seed)
        out = args.out_dir / f'r2_{args.tag}_seed{seed}.csv'
        with out.open('w', newline='') as fh:
            fh.write('Time,Operation,OrderId,Side,Volume,Price,Lifespan\n')
            fh.write('\n'.join(rows)); fh.write('\n')
        out.with_suffix('.params.json').write_text(json.dumps({**p, 'seed': seed}, indent=2))
        out.with_suffix('.truth.json').write_text(json.dumps(truth))
        print(f'{out}: {len(rows):,} rows, {len(truth["onsets"])} shocks, '
              f'{sum(b - a for a, b, v in truth["segs"] if v):.0f} s volatile')


if __name__ == '__main__':
    main()
