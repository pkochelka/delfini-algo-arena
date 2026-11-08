"""Round 4: a live forecast of the filter mean's future change, and whether trading on it pays.

    uv run python strategies/peter/r4_fwd.py logs/r4/fpath_round4_public.npz [--split 3000]

Sampled every 100 ms. Features (all causal): gap = F - market mid, exogenous / taker filter changes over the
trailing 0.5 / 2 s, EWMA of filled taker signs, P(volatile). Targets: F(t+h) - F(t), h = 0.5 / 1 / 2 / 5 / 10 s.
Per regime, OLS on t < split, scored out of sample on t >= split (R2, and the mean of the realised move in
deciles of the forecast). The last table is the trade check: buying at the ask (selling at the bid) whenever the
forecast clears the half spread + thr, marked at F(t+h).
"""
from __future__ import annotations

import argparse

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('path')
    ap.add_argument('--split', type=float, default=3000.0)
    ap.add_argument('--dt', type=float, default=0.1)
    a = ap.parse_args()
    d = dict(np.load(a.path))
    t = d['t'] * 1e-6
    t = t - t[0]
    F, q, pv = d['F'], d['q'], d['pv']
    dF = np.diff(F, prepend=F[0])
    isk = q != 0
    cX = np.cumsum(np.where(isk, 0.0, dF))
    cK = np.cumsum(np.where(isk, dF, 0.0))
    # EWMA of taker signs per event time, half-life 1 s
    g = np.arange(10.0, t[-1] - 11, a.dt)
    j = np.searchsorted(t, g, side='right') - 1
    bb, ba = d['bb'][j], d['ba'][j]
    mid = 0.5 * (bb + ba)
    ok = np.isfinite(mid)
    tq = t[isk]
    sq = np.sign(q[isk])
    cs = np.cumsum(sq)

    def flow(h):
        lo = np.searchsorted(tq, g - h)
        hi = np.searchsorted(tq, g, side='right')
        c0 = np.concatenate([[0.0], cs])
        return c0[hi] - c0[lo]

    def trail(c, h):
        k = np.maximum(np.searchsorted(t, g - h, side='right') - 1, 0)
        return c[j] - c[k]

    feats = {
        'gap': F[j] - mid,
        'X.5': trail(cX, 0.5), 'X2': trail(cX, 2.0),
        'K.5': trail(cK, 0.5), 'K2': trail(cK, 2.0),
        'fl1': flow(1.0), 'fl5': flow(5.0),
    }
    names = list(feats)
    Xm = np.column_stack([feats[n] for n in names])
    vol = pv[j] > 0.5
    tr = g < a.split
    for h in (0.5, 1.0, 2.0, 5.0, 10.0):
        k = np.searchsorted(t, g + h, side='right') - 1
        y = F[k] - F[j]
        print(f'==== h {h}s')
        for lab, m in (('calm', ~vol), ('volatile', vol)):
            A = np.column_stack([Xm, np.ones(len(g))])
            mtr, mte = m & tr & ok, m & ~tr & ok
            beta, *_ = np.linalg.lstsq(A[mtr], y[mtr], rcond=None)
            yh = A @ beta
            r = y[mte] - yh[mte]
            r2 = 1 - (r @ r) / ((y[mte] - y[mte].mean()) @ (y[mte] - y[mte].mean()))
            print(f'  {lab:8s} oos R2 {r2:+.3f}  ' + ' '.join(f'{n} {b:+.3f}' for n, b in zip(names, beta)))
            # gap-only model
            Ag = np.column_stack([feats['gap'], np.ones(len(g))])
            bg, *_ = np.linalg.lstsq(Ag[mtr], y[mtr], rcond=None)
            rg = y[mte] - (Ag @ bg)[mte]
            print(f'           gap-only: beta {bg[0]:+.3f}  oos R2 {1 - (rg @ rg) / ((y[mte] - y[mte].mean()) @ (y[mte] - y[mte].mean())):+.3f}')
            if h in (1.0, 5.0):
                qs = np.quantile(yh[mte], np.linspace(0, 1, 11))
                row = []
                for lo, hi in zip(qs[:-1], qs[1:]):
                    kk = mte & (yh >= lo) & (yh <= hi)
                    row.append(f'{yh[kk].mean():+.2f}->{y[kk].mean():+.2f}')
                print('           deciles pred->real: ' + '  '.join(row))
                # trade check: buy at ask when pred - (ask - F) > thr, mark at F(t+h)
                for thr in (0.0, 0.5, 1.0, 2.0):
                    eb = yh - (ba - F[j])
                    es = -yh - (F[j] - bb)
                    kb = mte & (eb > thr)
                    ks = mte & (es > thr)
                    pnl = np.concatenate([(F[k] - ba)[kb], (bb - F[k])[ks]])
                    print(f'           trade thr {thr}: n {pnl.size:6d} ({pnl.size / max(mte.sum(), 1):.3f} of samples)  '
                          f'mean edge at +{h}s {pnl.mean() if pnl.size else 0:+.2f}')


if __name__ == '__main__':
    main()
