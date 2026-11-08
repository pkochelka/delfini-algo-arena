"""Per-episode attribution of the d_agents overlay on the full round-4 tape.

uv run python strategies/adam/r4/agents_attrib.py '{"ag_r15":1}' [--post 10]
Runs d_agents with {} (= r4_best) and with the given overlay, records our position / PnL every event, finds the
TWAP and 15-lot episodes the strategy detected online, and prints per episode: base PnL and overlay PnL over
[detection, end + post], the base's position at detection, and the overlay's delta.
"""
import argparse, json, pathlib, sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
S = 'strategies/adam/r4/d_agents.py'
TAPE = 'data/round4_public.csv'


def run(params):
    from fastsim.engine import grader_config, run_backtest, load_strategy
    params = dict(params)
    # detectors on in both runs (detection never changes trading unless ag_tw / ag_r15 are set)
    params.setdefault('tw_on', 1)
    st = load_strategy(S, params)
    trace = []  # (t, pos, ag_dir, tw_n, r15_on)
    oq = st._quote

    def wq(f, bb, ba, bids, asks):
        oq(f, bb, ba, bids, asks)
        trace.append((st.now_t, st.pos, st.ag_dir, st.tw_n, int(st.r15_on), st.tw_s, st.r15_s, int(st.turb)))
    st._quote = wq
    r, curve = run_backtest(st, ROOT / TAPE, config=grader_config())
    return r.pnl, curve, np.array(trace, dtype=float)


def episodes(tr, col, side_col):
    on = tr[:, col] >= (2 if col == 3 else 1)
    eps, i, n = [], 0, len(on)
    idx = np.flatnonzero(np.diff(on.astype(int)) != 0) + 1
    starts = ([0] if on[0] else []) + [k for k in idx if on[k]]
    for s in starts:
        e = s
        while e < n and on[e]:
            e += 1
        eps.append((tr[s, 0], tr[min(e, n - 1), 0], int(tr[s, side_col]), tr[s, 1], int(tr[s, 7])))
    return eps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cfg')
    ap.add_argument('--post', type=float, default=10.0)
    a = ap.parse_args()
    cfg = json.loads(a.cfg)
    with ProcessPoolExecutor(2) as ex:
        fb, fo = ex.submit(run, {}), ex.submit(run, cfg)
        (pb, cb, tb), (po, co, to) = fb.result(), fo.result()
    cb, co = np.array(cb), np.array(co)

    def pnl_at(c, t):
        i = np.searchsorted(c[:, 0], t, side='right') - 1
        return c[max(i, 0), 1]
    print(f'total base {pb / 1e3:.1f}k overlay {po / 1e3:.1f}k delta {(po - pb) / 1e3:+.1f}k')
    for name, col, sc in (('TWAP', 3, 5), ('r15', 4, 6)):
        eps = episodes(tb, col, sc)
        print(f'\n{name}: {len(eps)} episodes (base-run detection)')
        print(f"{'t0':>8} {'dur':>5} {'side':>4} {'reg':>3} {'pos0':>5} {'base':>8} {'over':>8} {'delta':>8}")
        tot = []
        for t0, t1, s, p0, turb in eps:
            w0, w1 = t0, t1 + a.post
            db = pnl_at(cb, w1) - pnl_at(cb, w0)
            do = pnl_at(co, w1) - pnl_at(co, w0)
            tot.append((db, do))
            print(f'{t0:8.1f} {t1 - t0:5.1f} {s:+4d} {"V" if turb else "C":>3} {p0:+5.0f} {db:+8.0f} {do:+8.0f} {do - db:+8.0f}')
        if tot:
            x = np.array(tot)
            d = x[:, 1] - x[:, 0]
            print(f'sum base {x[:, 0].sum():+.0f} overlay {x[:, 1].sum():+.0f} delta {d.sum():+.0f} '
                  f'median delta {np.median(d):+.0f} won {(d > 0).sum()}/{len(d)}')


if __name__ == '__main__':
    main()
