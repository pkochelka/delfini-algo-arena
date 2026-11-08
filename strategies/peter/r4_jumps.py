"""Round-4 exogenous jumps of the hidden price (filter path from r4_fpath.py).

    uv run python strategies/peter/r4_jumps.py logs/r4/fpath_round4_public.npz [--thr 2.0] [--bin 0.05]

A jump = a bin whose exogenous filter change X (changes at non-taker events) exceeds thr ticks. Reports the jump
count by regime, size law, the sign agreement with the previous jump as a function of the gap, the drift of F in
the seconds after a jump (net of the jump), and whether the taker flow before a jump predicts its sign.
"""
from __future__ import annotations

import argparse

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('path')
    ap.add_argument('--thr', type=float, default=2.0)
    ap.add_argument('--bin', type=float, default=0.05)
    a = ap.parse_args()
    d = dict(np.load(a.path))
    t = d['t'] * 1e-6
    t = t - t[0]
    F, q, pv = d['F'], d['q'], d['pv']
    dF = np.diff(F, prepend=F[0])
    b = (t // a.bin).astype(np.int64)
    n = b[-1] + 1
    X = np.bincount(b, weights=np.where(q != 0, 0.0, dF), minlength=n)
    K = np.bincount(b, weights=np.where(q != 0, dF, 0.0), minlength=n)
    S = np.bincount(b, weights=np.sign(q), minlength=n)
    last = np.maximum(np.searchsorted(b, np.arange(n), side='right') - 1, 0)
    Fb, pvb = F[last], pv[last]
    J = np.flatnonzero(np.abs(X) > a.thr)
    # merge adjacent bins into one jump
    keep = [J[0]] if J.size else []
    for j in J[1:]:
        if j - keep[-1] > 2:
            keep.append(j)
    J = np.array(keep)
    size = np.array([X[j:j + 3].sum() for j in J])
    tj = J * a.bin
    vol = pvb[np.maximum(J - 1, 0)] > 0.5
    print(f'jumps |X| > {a.thr} per {a.bin}s bin: {J.size}  ({J.size / (n * a.bin) * 60:.1f}/min), volatile share {vol.mean():.2f}')
    print(f'  |size| quantiles 10/50/90/99: {np.round(np.quantile(np.abs(size), [.1, .5, .9, .99]), 1)}  up share {(size > 0).mean():.2f}')
    gap = np.diff(tj)
    agree = np.sign(size[1:]) == np.sign(size[:-1])
    print('  P(same sign as previous jump) by gap to it:')
    for lo, hi in ((0, 0.5), (0.5, 1), (1, 2), (2, 5), (5, 10), (10, 30), (30, 1e9)):
        m = (gap >= lo) & (gap < hi)
        if m.sum() > 5:
            print(f'    gap [{lo:5.1f},{hi:5.1f}) n {m.sum():5d}  same {agree[m].mean():.3f}')
    k = np.arange(1, 4)
    for lag in k:
        if size.size > lag:
            print(f'  sign ACF lag {lag}: {np.corrcoef(np.sign(size[lag:]), np.sign(size[:-lag]))[0, 1]:+.3f}')
    # after-jump drift (net of the jump bins), split by regime, in the direction of the jump
    print('  signed F drift after the jump (from +0.15 s), split exogenous X / taker K, and taker sign sum:')
    for hs in (0.5, 1, 2, 5, 10, 20):
        h = int(hs / a.bin)
        ok = J + 3 + h < n
        jj, sg = J[ok], np.sign(size[ok])
        cx = np.cumsum(X)
        ck = np.cumsum(K)
        cs = np.cumsum(S)
        dx = (cx[jj + 2 + h] - cx[jj + 2]) * sg
        dk = (ck[jj + 2 + h] - ck[jj + 2]) * sg
        ds = (cs[jj + 2 + h] - cs[jj + 2]) * sg
        vm = vol[ok]
        print(f'    +{hs:4.1f}s  X {dx.mean():+6.2f}  K {dk.mean():+6.2f}  takers {ds.mean():+5.2f}'
              f'   | volatile X {dx[vm].mean():+6.2f} K {dk[vm].mean():+6.2f}   calm X {dx[~vm].mean():+6.2f} K {dk[~vm].mean():+6.2f}')
    # same-direction continuation as a function of |size|
    h = int(5 / a.bin)
    ok = J + 3 + h < n
    cF = (Fb[np.minimum(J + 2 + h, n - 1)] - Fb[J + 2]) * np.sign(size)
    print('  F drift 0.15 -> 5 s after the jump by |size|:')
    for lo, hi in ((2, 4), (4, 8), (8, 16), (16, 1e9)):
        m = ok & (np.abs(size) >= lo) & (np.abs(size) < hi)
        if m.sum() > 3:
            print(f'    |J| [{lo},{hi}) n {m.sum():4d}  drift {cF[m].mean():+6.2f}  sd {cF[m].std():5.2f}')
    # do takers anticipate the jump?
    cs = np.cumsum(S)
    for hs in (0.25, 1, 3):
        h = int(hs / a.bin)
        ok = J - h > 0
        pre = (cs[J[ok] - 1] - cs[J[ok] - 1 - h]) * np.sign(size[ok])
        print(f'  taker sign sum in the {hs}s before the jump, signed by the jump: mean {pre.mean():+.3f} (n {ok.sum()})')
    # jump-run structure: runs of same-sign jumps with gaps < 10 s
    runs, cur = [], 1
    for g, ag in zip(gap, agree):
        if g < 10 and ag:
            cur += 1
        else:
            runs.append(cur)
            cur = 1
    runs.append(cur)
    runs = np.array(runs)
    print(f'  same-sign runs (gap < 10 s): mean length {runs.mean():.2f}, share of runs >= 3: {(runs >= 3).mean():.2f}')


if __name__ == '__main__':
    main()
