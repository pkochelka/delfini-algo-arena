"""Real vs synthetic round-3 tape: do the generator laws that turbotrader's optional features encode hold on the real tape?

    uv run python strategies/peter/tt_realsynth.py data/round3_public.csv data/synth/r3_tt_seed201.csv ...

Synthetic tapes (marek/research/round3/strategy/r3_synth.py) rated r3_grandpa's filter features +233k non-shock on
12 fresh draws, while the real public tape gave a fraction of that and the test set none, so the synth generator
must differ from the real one exactly where those features look. Two checks, per regime (real: Marek's 1 s Viterbi
regimes, flow/regimes_1s.csv; synth: the .truth.json segments):
    takers   IOC limit minus the opposite touch before the insert (buys; sells mirrored): the taker-reach feature
             (tk_on) assumes limit = round(S +- 1) calm / round(S +- 2.55) volatile plus a small tail
    ladder   maker inserts 2-6 ticks behind the same-side touch: rate, the share that start or continue an exact
             2000 us chain (one tick further out), and the size law of the rest ("singles"), which the ladder
             mixture (lm_on) models as LN(5.02, 0.37) calm / LN(3.52, 0.34) volatile at 5.5 / 8.2 per s
"""
from __future__ import annotations

import bisect
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analysis.cache import load_tape  # noqa: E402


def regimes(path: Path):
    tj = path.with_suffix('.truth.json')
    if tj.exists():
        return [(a, b, 'vol' if v else 'calm') for a, b, v in json.loads(tj.read_text())['segs']]
    f = ROOT / 'marek/research/round3/flow/regimes_1s.csv'
    return [(float(r['start']), float(r['end']), 'vol' if r['regime'] == 'volatile' else 'calm')
            for r in csv.DictReader(open(f))]


def label(ts: np.ndarray, segs) -> np.ndarray:
    starts = [a for a, b, g in segs]
    out = np.full(ts.size, 'calm', dtype=object)
    for j, t in enumerate(ts):
        i = bisect.bisect_right(starts, t) - 1
        if i >= 0 and t < segs[i][1]:
            out[j] = segs[i][2]
    return out


def main() -> None:
    for arg in sys.argv[1:]:
        path = Path(arg)
        tape = load_tape(path, verbose=False)
        ev, o = tape.events, tape.orders
        t0 = tape.start_us
        segs = regimes(path)
        dur = {g: sum(min(b, tape.duration_s) - a for a, b, gg in segs if gg == g and a < tape.duration_s)
               for g in ('calm', 'vol')}
        bid = ev.bid.to_numpy()
        ask = ev.ask.to_numpy()
        row = o.row.to_numpy()
        pb = np.where(row > 0, bid[np.maximum(row - 1, 0)], 0)
        pa = np.where(row > 0, ask[np.maximum(row - 1, 0)], 0)
        is_b = (o.side == 'B').to_numpy()
        px = o.price.to_numpy().astype(np.int64)
        ioc = o.ioc.to_numpy()
        ts = (o.t_us.to_numpy() - t0) / 1e6
        reg = label(ts, segs)
        print(f'== {path.name}: {tape.duration_s:.0f} s, calm {dur["calm"]:.0f} s, vol {dur["vol"]:.0f} s')
        # ---- takers: limit - opposite touch before (ticks through the touch)
        ok = ioc & (pb > 0) & (pa > 0)
        x = np.where(is_b, px - pa, pb - px)[ok]
        for g in ('calm', 'vol'):
            xs = x[reg[ok] == g]
            h = {k: np.mean(xs == k) for k in range(-1, 4)}
            print(f'  takers {g:4}: n {xs.size:6d}  limit - opp touch: <=-2 {np.mean(xs <= -2):.3f} '
                  + ' '.join(f'{k:+d} {v:.3f}' for k, v in h.items()) + f'  >=4 {np.mean(xs >= 4):.3f}')
        # ---- ladder: deep inserts 2..6 behind the same-side touch, chains of exact 2000 us rungs
        mk = ~ioc
        dist = o.dist.to_numpy()
        tu = o.t_us.to_numpy()
        key = set(zip(is_b[mk].tolist(), tu[mk].tolist(), px[mk].tolist()))

        def has(b, t, p):
            return any((b, t + d, p) in key for d in range(-3, 4))

        deep = mk & (dist <= -2) & (dist >= -6)
        idx = np.nonzero(deep)[0]
        nxt = np.array([has(is_b[i], tu[i] + 2000, px[i] - 1 if is_b[i] else px[i] + 1) for i in idx], dtype=bool)
        prv = np.array([has(is_b[i], tu[i] - 2000, px[i] + 1 if is_b[i] else px[i] - 1) for i in idx], dtype=bool)
        single = ~nxt & ~prv
        vol = o.volume.to_numpy()[idx].astype(float)
        for g in ('calm', 'vol'):
            m = reg[idx] == g
            if not m.any() or dur[g] <= 0:
                continue
            z = np.log(np.maximum(vol[m & single], 1))
            q = np.quantile(z, [0.1, 0.5, 0.9]) if z.size else [np.nan] * 3
            big = np.mean(z > 4.0) if z.size else np.nan
            print(f'  ladder {g:4}: deep {m.sum() / dur[g]:5.1f}/s, chain first {np.sum(m & nxt & ~prv) / dur[g]:4.2f}/s, '
                  f'rungs {np.sum(m & prv) / dur[g]:4.2f}/s, singles {np.sum(m & single) / dur[g]:5.1f}/s; '
                  f'singles log-size q10/50/90 {q[0]:.2f} {q[1]:.2f} {q[2]:.2f}, share > e^4 (55 lots) {big:.2f}')


if __name__ == '__main__':
    main()
