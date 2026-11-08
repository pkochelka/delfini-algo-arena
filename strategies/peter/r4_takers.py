"""Round 4: what sets a market taker's side? (filter path from r4_fpath.py)

    uv run python strategies/peter/r4_takers.py logs/r4/fpath_round4_public.npz

One row per filled market taker event. Linear probability model of its side (+1 buy / -1 sell) on
  dF_h     the filter mean's change over the trailing h s (h = 0.2, 1, 3, 10), split into the exogenous part X
           (changes at non-taker events) and the taker part K
  s1, s10  the previous taker's side, and the mean side of the previous 10 takers
  gap      F - mid just before it
by regime. Also the taker rate against |recent move| and the side probability binned by the trailing 1 s move.
"""
from __future__ import annotations

import sys

import numpy as np


def main(path: str) -> None:
    d = dict(np.load(path))
    t = d['t'] * 1e-6
    F, q, pv = d['F'], d['q'], d['pv']
    dF = np.diff(F, prepend=F[0])
    isk = q != 0
    cX = np.cumsum(np.where(isk, 0.0, dF))
    cK = np.cumsum(np.where(isk, dF, 0.0))
    mid = 0.5 * (d['bb'] + d['ba'])
    idx = np.flatnonzero(isk)
    idx = idx[(t[idx] - t[0] > 15)]
    sg = np.sign(q[idx])
    pre = idx - 1  # state just before the taker event

    def trail(c, h):
        j = np.searchsorted(t, t[pre] - h) - 1
        j = np.maximum(j, 0)
        return c[pre] - c[j]

    cols, names = [], []
    for h in (0.2, 1.0, 3.0, 10.0):
        cols += [trail(cX, h), trail(cK, h)]
        names += [f'X{h:g}', f'K{h:g}']
    s1 = np.concatenate([[0.0], sg[:-1]])
    cs = np.cumsum(np.concatenate([[0.0], sg]))
    s10 = np.zeros(sg.size)
    s10[10:] = (cs[10:-1] - cs[:-11]) / 10
    gap = F[pre] - mid[pre]
    cols += [s1, s10, gap]
    names += ['s1', 's10', 'gap']
    vol = pv[pre] > 0.5
    for lab, m in (('calm', ~vol), ('volatile', vol)):
        A = np.column_stack([c[m] for c in cols] + [np.ones(m.sum())])
        ok = np.all(np.isfinite(A), axis=1)
        A, y = A[ok], sg[m][ok]
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        r = y - A @ beta
        se = np.sqrt(np.diag(r @ r / (y.size - A.shape[1]) * np.linalg.inv(A.T @ A)))
        print(f'--- {lab}: n {y.size}, buy share {(y > 0).mean():.3f}, R2 {1 - r @ r / ((y - y.mean()) @ (y - y.mean())):.4f}')
        print('   ' + '  '.join(f'{n} {b:+.4f}({b / s:+.1f})' for n, b, s in zip(names, beta, se)))
        # side probability by trailing 1 s total move
        mv = (cols[2] + cols[3])[m][ok]
        print('   P(buy) by trailing 1 s move of F:')
        for lo, hi in ((-1e9, -8), (-8, -3), (-3, -1), (-1, -0.2), (-0.2, 0.2), (0.2, 1), (1, 3), (3, 8), (8, 1e9)):
            k = (mv >= lo) & (mv < hi)
            if k.sum() > 30:
                print(f'     [{lo:6.1f},{hi:6.1f}) n {k.sum():6d}  P(buy) {(y[k] > 0).mean():.3f}')
        mv = gap[m][ok]
        print('   P(buy) by gap F - mid:')
        for lo, hi in ((-1e9, -2), (-2, -1), (-1, -0.3), (-0.3, 0.3), (0.3, 1), (1, 2), (2, 1e9)):
            k = (mv >= lo) & (mv < hi)
            if k.sum() > 30:
                print(f'     [{lo:6.1f},{hi:6.1f}) n {k.sum():6d}  P(buy) {(y[k] > 0).mean():.3f}')


if __name__ == '__main__':
    for p in sys.argv[1:]:
        print('====', p)
        main(p)
