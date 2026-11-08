"""Exact regime segmentation: 2-state Viterbi on per-insert llr (vol vs calm, from switch3.py) with switch penalty PEN.
Usage: viterbi_seg.py TAG [PEN]. Writes vol_segments.csv (start,end,dur,seg_llr,gain_on,gain_off)."""
import sys
import numpy as np, pandas as pd
from numba import njit
import prep
tag = sys.argv[1]; PEN = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
d = prep.build(); t = d['t']; llr = np.load(prep.common.CACHE / f'llr_{tag}.npy')


@njit(cache=True)
def vit(l, pen):
    n = len(l); B = np.zeros((n, 2), np.int8); v0 = 0.0; v1 = -pen
    for i in range(n):
        # stay or switch
        a0, b0 = v0, v1 - pen
        a1, b1 = v1, v0 - pen
        if a0 >= b0: n0 = a0; B[i, 0] = 0
        else: n0 = b0; B[i, 0] = 1
        if a1 >= b1: n1 = a1; B[i, 1] = 1
        else: n1 = b1; B[i, 1] = 0
        v0 = n0; v1 = n1 + l[i]
    s = np.zeros(n, np.int8); s[n - 1] = 0 if v0 >= v1 else 1
    for i in range(n - 1, 0, -1):
        s[i - 1] = B[i, s[i]]
    return s


s = vit(llr, PEN)
ch = np.flatnonzero(np.diff(s)) + 1
on = ch[s[ch] == 1]; off = ch[s[ch] == 0]
if s[0] == 1: on = np.r_[0, on]
if s[-1] == 1: off = np.r_[off, len(s)]
rows = []
for a, b in zip(on, off):
    # evidence: llr sum inside, and gain within 5 s outside
    rows.append(dict(start=t[a], end=t[b] if b < len(t) else 6000.0, seg_llr=llr[a:b].sum(),
                     pre5=llr[np.searchsorted(t, t[a] - 5):a].sum(), post5=llr[b:np.searchsorted(t, (t[b] if b < len(t) else 6000) + 5)].sum()))
X = pd.DataFrame(rows); X['dur'] = X.end - X.start
pd.set_option('display.width', 200); pd.set_option('display.max_rows', 200)
print(X.round(3).to_string())
print('n', len(X), 'vol frac %.4f' % (X.dur.sum() / 6000), 'mean vol dwell %.1f' % X.dur.mean())
gaps = X.start.to_numpy()[1:] - X.end.to_numpy()[:-1]
print('calm dwell mean %.1f min %.2f; vol dwell min %.2f max %.1f' % (gaps.mean(), gaps.min(), X.dur.min(), X.dur.max()))
X.to_csv(prep.common.HERE / 'vol_segments.csv', index=False)
np.save(prep.common.CACHE / f'state_{tag}.npy', s)
