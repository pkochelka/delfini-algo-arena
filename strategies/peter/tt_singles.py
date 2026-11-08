"""Fit the hidden ladder-single rate per regime: deep (2-6 behind the same-side touch) non-chain maker inserts are a
mixture of base makers (size law fitted on the shallow inserts, dist >= 0) and ladder singles LN(mu, sd); EM on
log-size with the base law fixed. Usage: uv run python strategies/peter/tt_singles.py TAPE [TAPE ...]"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tt_realsynth import load_tape, regimes, label  # noqa: E402

def npdf(z, m, s):
    return np.exp(-0.5 * ((z - m) / s) ** 2) / s

for arg in sys.argv[1:]:
    path = Path(arg); tape = load_tape(path, verbose=False); o = tape.orders
    segs = regimes(path)
    dur = {g: sum(min(b, tape.duration_s) - a for a, b, gg in segs if gg == g and a < tape.duration_s) for g in ('calm', 'vol')}
    mk = ~o.ioc.to_numpy(); dist = o.dist.to_numpy(); is_b = (o.side == 'B').to_numpy()
    px = o.price.to_numpy().astype(np.int64); tu = o.t_us.to_numpy()
    reg = label((tu - tape.start_us) / 1e6, segs)
    z_all = np.log(np.maximum(o.volume.to_numpy().astype(float), 1))
    key = set(zip(is_b[mk].tolist(), tu[mk].tolist(), px[mk].tolist()))
    has = lambda b, t, p: any((b, t + d, p) in key for d in range(-3, 4))
    print(f'== {path.name}')
    for g in ('calm', 'vol'):
        sh = mk & (dist >= 0) & (reg == g)
        bm, bs = z_all[sh].mean(), z_all[sh].std()
        idx = np.nonzero(mk & (dist <= -2) & (dist >= -6) & (reg == g))[0]
        ch = np.array([has(is_b[i], tu[i] + 2000, px[i] - 1 if is_b[i] else px[i] + 1) or
                       has(is_b[i], tu[i] - 2000, px[i] + 1 if is_b[i] else px[i] - 1) for i in idx], dtype=bool)
        z = z_all[idx[~ch]]
        w, m, s = 0.2, (5.0 if g == 'calm' else 3.5), 0.4
        for _ in range(200):
            a = w * npdf(z, m, s); b = (1 - w) * npdf(z, bm, bs); r = a / (a + b)
            w = r.mean(); m = (r * z).sum() / r.sum(); s = max(0.1, np.sqrt((r * (z - m) ** 2).sum() / r.sum()))
        print(f'  {g:4}: base LN({bm:.2f},{bs:.2f}); deep non-chain {z.size / dur[g]:5.1f}/s -> ladder singles '
              f'{w * z.size / dur[g]:4.2f}/s LN({m:.2f},{s:.2f}) (share {w:.3f})')
