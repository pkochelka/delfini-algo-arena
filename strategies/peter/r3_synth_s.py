"""True S path of Marek's synthetic round-3 tapes: re-runs marek/research/round3/strategy/r3_synth.py's generator (same seed
-> same tape) with one injected line that records (t, S, permanent part Fp) before every candidate event.

    uv run python strategies/peter/r3_synth_s.py --seeds 1 2 3 4 5 6     # -> data/synth/r3_v1_seed<k>.spath.npy
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'marek/research/round3/strategy/r3_synth.py'

src = SRC.read_text()
a = "        t_prev = t\n        kind = k_all[i]\n"
assert src.count(a) == 1
src = src.replace(a, "        t_prev = t\n        _SP.append((t, S, Fp))\n        kind = k_all[i]\n")
src = src.replace("if __name__ == '__main__':", "if False:")
g = {'__name__': 'r3_synth_s'}
exec(compile(src, str(SRC), 'exec'), g)

ap = argparse.ArgumentParser(); ap.add_argument('--seeds', type=int, nargs='+'); ap.add_argument('--tag', default='v1')
a = ap.parse_args()
for seed in a.seeds:
    g['_SP'] = sp = []
    rows, truth = g['generate'](dict(g['PARAMS']), seed)
    arr = np.array(sp, dtype=np.float64)
    out = ROOT / f'data/synth/r3_{a.tag}_seed{seed}.spath.npy'
    np.save(out, arr)
    print(out, arr.shape, 'rows', len(rows))
