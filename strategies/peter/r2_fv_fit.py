"""Offline fit / check of the fair-value filter (fv_mm.FV) on a raw tape, without the simulator.

    uv run python strategies/peter/r2_fv_fit.py data/round2_public.csv --t0 0 --t1 600 --set sigma=0.3 lam=0.45
    uv run python strategies/peter/r2_fv_fit.py data/round2_public.csv --grid sigma=0.2,0.4 rho0=0.02,0.1

Score: mean predictive log-likelihood of maker inserts (higher = better generator fit), plus how well
the filter mean predicts the future mid: R^2 of mid(t+H) - mid(t) on (mean - mid) at 100 ms samples.
"""
from __future__ import annotations

import argparse, itertools, json, math, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis.cache import load_tape  # noqa: E402
from fv_mm import FV, FV_PARAMS  # noqa: E402

_EV = None


def _load(csv):
    global _EV
    if _EV is None:
        t = load_tape(csv, verbose=False)
        ev, o = t.events, t.orders
        filled = np.zeros(len(ev), dtype=np.int64)
        oo = o[o.ioc]
        filled[oo.row.to_numpy()] = oo.filled.to_numpy()
        bid = ev.bid.to_numpy(); ask = ev.ask.to_numpy()
        _EV = dict(T=ev.t_us.to_numpy() / 1e6, ins=(ev.op == 'I').to_numpy(), G=(ev.lifespan == 'G').to_numpy(),
                   B=(ev.side == 'B').to_numpy(), px=np.nan_to_num(ev.price.to_numpy()).astype(np.int64),
                   filled=filled, pb=np.r_[0, bid[:-1]], pa=np.r_[0, ask[:-1]], bid=bid, ask=ask)
    return _EV


def run(csv, over, t0, t1, keep=False):
    E = _load(csv)
    T, ins, G, B, px, filled, pb, pa = (E[k] for k in ('T', 'ins', 'G', 'B', 'px', 'filled', 'pb', 'pa'))
    p = FV_PARAMS | over
    i0 = int(np.searchsorted(T, t0)); i1 = int(np.searchsorted(T, t1))
    f = FV(p, (pb[i0] + pa[i0]) / 2 if pb[i0] and pa[i0] else 10000)
    warm = t0 + 20
    samp_t, samp_m, samp_sd = [], [], []
    nxt = warm
    ll0 = None
    for i in range(i0, i1):
        if not ins[i]:
            continue
        t = T[i]
        if ll0 is None and t >= warm:
            ll0 = (f.ll, f.n_ll)
        if G[i]:
            opp = pa[i] if B[i] else pb[i]
            f.maker(t, bool(B[i]), int(px[i]), int(opp) if opp else None)
        else:
            f.taker(t, bool(B[i]), int(px[i]), int(filled[i]))
        if t >= nxt:
            samp_t.append(t); samp_m.append(f.mean); samp_sd.append(f.sd)
            nxt = t + 0.1
    ll = (f.ll - ll0[0]) / max(1, f.n_ll - ll0[1])
    st = np.array(samp_t); sm = np.array(samp_m)
    j = np.searchsorted(T, st, 'right') - 1
    mid = (E['bid'] + E['ask']) / 2.0
    out = {'ll': ll, 'jumps': f.jumps, 'sd_med': float(np.median(samp_sd))}
    for H in (0.5, 2.0):
        jf = np.searchsorted(T, st + H, 'right') - 1
        ok = (E['bid'][j] > 0) & (E['ask'][j] > 0) & (E['bid'][jf] > 0) & (E['ask'][jf] > 0) & (st + H < t1)
        y = (mid[jf] - mid[j])[ok]; x = (sm - mid[j])[ok]
        b = (x @ y) / (x @ x)
        out[f'r2_{H}'] = 1 - ((y - b * x) ** 2).sum() / (y ** 2).sum()
        out[f'b_{H}'] = b
    if keep:
        out['samples'] = (st, sm, np.array(samp_sd))
    return out


def parse_set(items):
    d = {}
    for it in items:
        k, v = it.split('=')
        d[k] = float(v) if '.' in v or 'e' in v else int(v)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('csv'); ap.add_argument('--t0', type=float, default=0); ap.add_argument('--t1', type=float, default=600)
    ap.add_argument('--set', nargs='*', default=[]); ap.add_argument('--grid', nargs='*', default=[])
    ap.add_argument('--windows', default='', help='t0:t1,t0:t1 ... evaluated separately and averaged')
    ap.add_argument('--jobs', type=int, default=12)
    a = ap.parse_args()
    base = parse_set(a.set)
    keys = [g.split('=')[0] for g in a.grid]
    vals = [[float(x) for x in g.split('=')[1].split(',')] for g in a.grid]
    combos = [dict(zip(keys, c)) for c in itertools.product(*vals)] or [{}]
    wins = [tuple(map(float, w.split(':'))) for w in a.windows.split(',')] if a.windows else [(a.t0, a.t1)]
    st = time.time()
    with ProcessPoolExecutor(a.jobs) as ex:
        futs = [[ex.submit(run, a.csv, base | c, w0, w1) for (w0, w1) in wins] for c in combos]
        for c, fs in zip(combos, futs):
            rs = [f.result() for f in fs]
            agg = {k: float(np.mean([r[k] for r in rs])) for k in rs[0]}
            print(json.dumps(c), ' '.join(f'{k}={v:.4f}' for k, v in agg.items()),
                  '| per-window ll', ' '.join(f"{r['ll']:.4f}" for r in rs), flush=True)
    print(f'{time.time() - st:.0f}s')


if __name__ == '__main__':
    main()
