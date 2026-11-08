"""tt_synth (Peter, 2026-11-08): a calibrated copy of marek/research/round3/strategy/r3_synth.py for turbotrader's
selection tapes. Changes against Marek's generator, each to match the real round-3 tape (tt_realsynth.py checks):
  - bug fix: a ladder's length used the same uniform as its start offset (u[2]), so every multi-rung ladder not
    starting at touch-2 started at touch-6 and was cut to 2 rungs; the length now has its own draw (u[3])
  - TT_FIT below: Marek's own final real-tape fit (marek/research/round3/s_dynamics/params_final.json) where
    r3_synth still had older values (no residual volatile jumps, volatile impact 0.062, taker tails, ...), and
    the calibrated ladder / taker-tail values

Marek's original docstring:
Synthetic round-3 tapes = marek/research/round2/r2_synth.py + the round-3 changes (marek/research/round3/*/README.md):
    ladder maker  events at lad_rate /s (regime), side 50/50, length 1 with prob lad_single, else 2..6 (lad_len weights);
                  first rung at same-side touch -2 (prob lad_p2) or -3..-6, never past -7; rungs exactly 2000 us apart,
                  one tick further out each; one lognormal size per ladder (+ N(0, 0.15) log noise per rung); no ladders
                  in a drought. Cancel hazard by distance d behind the same-side touch, checked every 10 ms:
                  d >= 8: 45 /s, 5..7: 1 /s, 2..4: 0.05 /s, 1: 3 /s, <= 0: 5.5 /s.
    transient     S = F + Y. A share 1 - pperm of each fill's kick (and of xi if tr_xi) goes into Y, which decays with
                  time constant tdec (regime); shock jumps and residual jumps are permanent.
    regime        p_cv 0.0065, p_vc 0.0166 (r3 dwell 154 s / 60 s); impact and maker laws from s_dynamics/params_v3.json.
    burst         jump-side head (+head_rate /s for head_s), then +burst_side /s per side falling linearly to 0 at burst_end_s.

Usage:
    uv run python marek/research/round3/strategy/r3_synth.py --seeds 1 2 3 [--duration 6000] [--set vol_c=0.15 ...]

Writes data/synth/r3_<tag>_seed<k>.csv (+ .params.json and .truth.json with regime segments and shock onsets).

Round-2 model notes (unchanged):

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
    'calm_c': 0.0171, 'calm_e': 0.61, 'calm_xi': 0.121, 'calm_sig': 0.0,
    'calm_d_shift': 0.0487, 'calm_d_scale': 0.4756, 'calm_off': 1.0, 'calm_eps': 0.0245, 'calm_tau': 3.415,
    'calm_taker_rate': 4.0, 'calm_alpha': 1.73, 'calm_beta': 6.1, 'calm_lamj': 0.0, 'calm_bj': 1.0,
    'calm_pperm': 0.583, 'calm_tdec': 17.3,
    'calm_lad_rate': 6.6, 'calm_lad_single': 0.83, 'calm_lad_mu': 4.98, 'calm_lad_sd': 0.59,
    # volatile
    'vol_c': 0.075, 'vol_e': 0.606, 'vol_xi': 0.276, 'vol_sig': 0.0,
    'vol_d_shift': 0.69, 'vol_d_scale': 1.06, 'vol_off': 2.55, 'vol_eps': 0.07, 'vol_tau': 6.3,
    'vol_taker_rate': 6.0, 'vol_alpha': 1.9, 'vol_beta': 4.1, 'vol_lamj': 0.1, 'vol_bj': 0.84,
    'vol_pperm': 0.55, 'vol_tdec': 15.0,
    'vol_lad_rate': 19.4, 'vol_lad_single': 0.42, 'vol_lad_mu': 3.50, 'vol_lad_sd': 0.40,
    'p_cv': 0.0065, 'p_vc': 0.0166,
    'tr_xi': 1.0, 'lad_p2': 0.6, 'lad_rung_sd': 0.15,
    'lad_h8': 45.0, 'lad_h5': 1.0, 'lad_h2': 0.05, 'lad_h1': 3.0, 'lad_h0': 5.5,
    'vol_lad_h5': -1.0,  # >= 0: separate volatile hazard at touch-5..-7 (makers README: calm 0.15-0.33, vol 0.8-2.9)
    'lad_rung_u': 0.0,  # > 0: rung size B * U(1 - u, 1 + u) instead of lognormal rung noise (makers README: u = 0.25)
    'head_s': 0.2, 'head_rate': 20.0, 'burst_side': 18.0,
    # shocks
    'shocks': 1.0,
    'gap_lo': 150.0, 'gap_hi': 300.0,
    'drought_s': 3.0, 'drought_mult': 0.30, 'drought_alpha': 6.3,
    'drought_d_shift': 0.54, 'drought_d_scale': 1.33,
    'jump_sd': 47.0,
    'burst_rate': 0.0, 'burst_flat_s': 1.25, 'burst_end_s': 3.0,
    'first_id': 1000001,
}


# ---- tt_synth calibration (2026-11-08). params_final (Marek's final real-tape fit): calm c / e / xi / pperm / tdec,
# volatile c / e / xi / pperm / tdec / D law / taker tail, no residual volatile jumps (lamj 0.0001).
TT_FIT = {
    'calm_c': 0.01677, 'calm_e': 0.616, 'calm_xi': 0.130, 'calm_pperm': 0.50, 'calm_tdec': 16.7,
    'vol_c': 0.0619, 'vol_e': 0.628, 'vol_xi': 0.261, 'vol_pperm': 0.553, 'vol_tdec': 13.6,
    'vol_d_shift': 0.673, 'vol_d_scale': 1.0, 'vol_eps': 0.046, 'vol_tau': 5.93, 'vol_lamj': 0.0001,
    'vol_lad_tilt': 0.0,
}
# calibrated on the real tape with tt_realsynth.py (real vs these tapes, per regime): ladder chain starts / rungs per s
# calm 1.15 / 1.46 (r3_synth after the bug fix: 0.73 / 0.95), volatile 10.4 / 18.6 (7.4 / 9.7); calm taker limits
# +2 / +3 / >=4 ticks through the opposite touch 4.1 / 1.2 / 2.7 % (r3_synth 3.4 / 0.6 / 1.5 %); calm D law of the
# makers README (0.066 + Exp(0.450)). Not matched: volatile takers <= -2 ticks inside the touch (real 3.4 %, here
# ~8.5 %) and volatile deep singles (real 75 /s, here ~90 /s); vol_beta 6.1 did not move them, and the real tape's
# volatile labels (1 s Viterbi) mix in calm-like seconds, so part of the gap is the labelling.
TT_FIT.update({'calm_lad_single': 0.73, 'vol_lad_single': 0.18, 'vol_lad_tilt': 0.5,
               'calm_eps': 0.045, 'calm_tau': 2.8, 'calm_d_shift': 0.066, 'calm_d_scale': 0.450})
PARAMS.update(TT_FIT)

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
    mr = p['maker_rate']
    tr_max = max(p['calm_taker_rate'], p['vol_taker_rate']) + p['burst_rate'] / 2 + p['burst_side'] + p['head_rate']
    lj_max = max(p['calm_lamj'], p['vol_lamj'])
    lad_max = max(p['calm_lad_rate'], p['vol_lad_rate'])
    times, kinds = [], []
    for kind, rate in ((0, mr), (1, tr_max), (2, tr_max), (3, lj_max), (4, lad_max)):
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

    lad_w = [12.1, 6.3, 2.8, 1.5, 1.2]
    lad_n = rng.choice(np.arange(2, 7), size=n, p=np.array(lad_w) / sum(lad_w)).tolist()
    # tt_synth: volatile ladders are longer on the real tape (1.78 rungs per chain vs 1.27 calm): length weights
    # tilted by exp(vol_lad_tilt * (L - 2)) in the volatile regime
    lw_v = np.array(lad_w) * np.exp(p.get('vol_lad_tilt', 0.0) * np.arange(5))
    lad_n_v = rng.choice(np.arange(2, 7), size=n, p=lw_v / lw_v.sum()).tolist()
    lad_u = rng.random((n, 4)).tolist()  # tt_synth: u[3] = the single-or-chain draw (was u[2], shared with d0)
    lad_z = rng.normal(0, 1, (n, 7)).tolist()
    st = {'vol': segs[0][2], 'drought': False, 'burst_t0': -1e9, 'jdir': 0}
    S = p['s0']
    Fp = S  # permanent part; S = Fp + Y
    Y = 0.0
    S_rr = S  # S at the last rerate
    t_dec = 0.0
    rungs: list = []  # heap (t, k, side, px, vol)
    lad_live: dict[int, tuple] = {}  # oid -> (side, px)
    lad_next = 0.0
    hz_u = np.random.default_rng(seed + 7777)
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

    def lad_insert(t, s, px, v):
        nonlocal oid
        rows.append(f'{stamp(t)},Insert,{oid},{"B" if s == 0 else "A"},{v},{px},G')
        lv = levels[s]
        if px not in lv:
            lv[px] = deque(); qty[s][px] = 0; insort(prices[s], px)
        entry = [oid, v]
        lv[px].append(entry); qty[s][px] += v
        live[oid] = (s, px, entry)
        lad_live[oid] = (s, px)
        oid += 1

    def lad_rungs(t):
        while rungs and rungs[0][0] <= t:
            tr, _, s, px, v = heapq.heappop(rungs)
            if st['drought'] or px <= 0:
                continue
            if prices[1 - s]:  # never cross the other side
                if (s == 0 and px >= prices[1][0]) or (s == 1 and px <= prices[0][-1]):
                    continue
            lad_insert(tr, s, px, v)

    def lad_cancels(t):
        nonlocal lad_next
        if t < lad_next:
            return
        dt = 0.01 if lad_next else 0.0
        lad_next = t + 0.01
        if not lad_live or not dt:
            return
        bb = prices[0][-1] if prices[0] else None
        ba = prices[1][0] if prices[1] else None
        drop = []
        for o, (s, px) in lad_live.items():
            if o not in live:
                drop.append(o); continue
            tp = bb if s == 0 else ba
            if tp is None:
                continue
            d = (tp - px) if s == 0 else (px - tp)
            h5 = p['vol_lad_h5'] if (st['vol'] and p['vol_lad_h5'] >= 0) else p['lad_h5']
            h = p['lad_h8'] if d >= 8 else h5 if d >= 5 else p['lad_h2'] if d >= 2 else p['lad_h1'] if d == 1 else p['lad_h0']
            if hz_u.random() < h * dt:
                drop.append(o)
                if o in live:
                    do_cancel(t, o)
        for o in drop:
            lad_live.pop(o, None)

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
                S += val; Fp += val
                st['burst_t0'] = tc
                st['jdir'] = 1 if val > 0 else -1
            rerate(tc)
        flush(t)
        lad_rungs(t)
        lad_cancels(t)
        if Y and t > t_dec:
            dY = Y * -math.expm1(-(t - t_dec) / p[pref() + 'tdec'])
            Y -= dY; S -= dY
        t_dec = t
        if abs(S - S_rr) > 0.05:
            S_rr = S
            rerate(t)
        sig = p[pref() + 'sig']
        if sig > 0 and t > t_prev:
            S += sig * math.sqrt(t - t_prev) * s_noise[i]
            if i % 50 == 0:
                rerate(t)
        t_prev = t
        kind = k_all[i]
        if kind == 3:
            lam = p[pref() + 'lamj']
            if st['drought'] or u_thin[i] > lam / lj_max:
                continue
            S += p[pref() + 'bj'] * j_lap[i]; Fp += p[pref() + 'bj'] * j_lap[i]
            S_rr = S
            rerate(t)
            continue
        if kind == 4:
            if st['drought'] or u_thin[i] > p[pref() + 'lad_rate'] / lad_max:
                continue
            u = lad_u[i]
            s = 0 if u[0] < 0.5 else 1
            tp = (prices[0][-1] if prices[0] else None) if s == 0 else (prices[1][0] if prices[1] else None)
            if tp is None:
                continue
            d0 = 2 if u[1] < p['lad_p2'] else 3 + min(3, int(u[2] * 4))
            L = 1 if u[3] < p[pref() + 'lad_single'] else (lad_n_v[i] if pref() == 'vol_' else lad_n[i])
            L = min(L, 8 - d0)
            z = lad_z[i]
            base = p[pref() + 'lad_mu'] + p[pref() + 'lad_sd'] * z[0]
            for k in range(L):
                if p['lad_rung_u'] > 0:
                    v = max(1, int(round(math.exp(base) * (1 + p['lad_rung_u'] * (2 * lad_u[i - k - 1][0] - 1)))))
                else:
                    v = max(1, int(round(math.exp(base + p['lad_rung_sd'] * z[k + 1]))))
                px = tp - (d0 + k) if s == 0 else tp + (d0 + k)
                heapq.heappush(rungs, (t + 0.002 * k, k, s, px, v))
            lad_rungs(t)
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
                if dtb < p['head_s']:
                    if (kind == 1) == (st['jdir'] > 0):
                        rate += p['head_rate']
                else:
                    rate += p['burst_side'] * (p['burst_end_s'] - dtb) / (p['burst_end_s'] - p['head_s'])
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
                kick = (c if buy else -c) * filled ** p[pref() + 'e']
                xi = p[pref() + 'xi'] * xi_noise[i]
                tr = 1 - p[pref() + 'pperm']
                dy = tr * (kick + (xi if p['tr_xi'] else 0.0))
                S += kick + xi; Y += dy; Fp += kick + xi - dy
                S_rr = S
                rerate(t)
    flush(T)
    truth = {'onsets': onsets, 'jumps': jumps, 'segs': segs}
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
        out = args.out_dir / f'tt_{args.tag}_seed{seed}.csv'
        with out.open('w', newline='') as fh:
            fh.write('Time,Operation,OrderId,Side,Volume,Price,Lifespan\n')
            fh.write('\n'.join(rows)); fh.write('\n')
        out.with_suffix('.params.json').write_text(json.dumps({**p, 'seed': seed}, indent=2))
        out.with_suffix('.truth.json').write_text(json.dumps(truth))
        print(f'{out}: {len(rows):,} rows, {len(truth["onsets"])} shocks, '
              f'{sum(b - a for a, b, v in truth["segs"] if v):.0f} s volatile')


if __name__ == '__main__':
    main()
