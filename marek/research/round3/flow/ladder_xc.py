"""Cross-intensity of ladder starts (by side) after same-side touch-price changes, split by direction
(touch improves = moves toward the other side; touch retreats), calm regime only, vs local baseline (±3–6 s)."""
import numpy as np, pandas as pd
from pathlib import Path
D = Path(__file__).parent; C = D / 'cache'
r = pd.read_pickle(C / 'rows.pkl'); L = pd.read_pickle(C / 'ladders.pkl'); R = pd.read_csv(D / 'regimes.csv')
t = r.t.to_numpy()
def inreg(x, reg):
    rr = R[R.regime == reg]; i = np.searchsorted(rr.start.to_numpy(), x) - 1
    return (i >= 0) & (x < rr.end.to_numpy()[np.maximum(i, 0)] - 6) & (x > rr.start.to_numpy()[np.maximum(i, 0)] + 6)
win = [(0, .002), (.002, .005), (.005, .01), (.01, .02), (.02, .05), (.05, .1), (.1, .2), (.2, .5), (.5, 1), (1, 2)]
print('windows', win)
for reg in ['calm', 'volatile']:
    for side, col, osd in [(0, 'bid_after', 'ask_after'), (1, 'ask_after', 'bid_after')]:
        px = r[col].to_numpy(); ch = np.flatnonzero(np.diff(px) != 0) + 1
        d = np.sign(px[ch] - px[ch - 1]) * (1 if side == 0 else -1)  # +1 = improves (bid up / ask down)
        lt = np.sort(L.t0[L.side == side].to_numpy()); lo = np.sort(L.t0[L.side != side].to_numpy())
        for nm, sel in [('improve', d > 0), ('retreat', d < 0)]:
            src = t[ch][sel]; src = src[inreg(src, reg)]
            for tgt_nm, tgt in [('same', lt), ('opp', lo)]:
                base = (np.searchsorted(tgt, src - 3) - np.searchsorted(tgt, src - 6) + np.searchsorted(tgt, src + 6) - np.searchsorted(tgt, src + 3)) / 6
                out = [round((np.searchsorted(tgt, src + b) - np.searchsorted(tgt, src + a, 'right')).sum() / (base.sum() * (b - a)), 2) for a, b in win]
                print(f'{reg:8s} side {side} touch {nm:7s} (n {len(src):6d}) -> ladder {tgt_nm}: {out}')
