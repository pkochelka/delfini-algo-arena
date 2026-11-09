"""Why did the simple sniper (not) fire? Runs r4_simple with overrides, logs every _fire call and every market
taker event near each shock's t_jump, and prints the chosen shocks.
uv run --with pyarrow python marek/research/round4/edge/tools/firelog.py 1 2 14 20 [--set k=v ...]"""
import argparse, ast, pathlib, sys
import numpy as np, pandas as pd
ROOT = pathlib.Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from fastsim.engine import load_strategy, run_backtest
from fastsim.tape import ensure_cache
R4 = pathlib.Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser(); ap.add_argument('ids', nargs='*', type=int); ap.add_argument('--set', nargs='*', default=[])
ap.add_argument('--strat', default='strategies/marek/round4/r4_simple.py'); ap.add_argument('--w', type=float, default=0.12)
a = ap.parse_args()
over = {k: ast.literal_eval(v) for k, v in (kv.split('=', 1) for kv in a.set)}
ev = pd.read_csv(R4 / 'flow/shock_events.csv')
tj = ev.t_jump.values
s = load_strategy(a.strat, over)
fires, events = [], []
orig_fire, orig_obc, orig_trig = s._fire, s.on_order_book_change, s._shock_trigger


def near(t):
    i = np.argmin(np.abs(tj - t))
    return i if abs(tj[i] - t) <= a.w else None


def fire(t, d, pbb, pba, edge=0.0):
    before = s.ss_dir
    orig_fire(t, d, pbb, pba, edge)
    fires.append((t, d, edge, pbb, pba, s.pos, before, s.ss_dir, s.onset, s.sg_t))


def trig(t, is_bid, price, pbb, pba):
    i = near(t)
    if i is not None:
        events.append((t, 'quote', 'B' if is_bid else 'A', price, pbb, pba, s.pos, s.ss_dir, s._in_window(t)))
    orig_trig(t, is_bid, price, pbb, pba)


def obc(snap, trades):
    t = snap.timestamp_us * 1e-6
    i = near(t)
    if i is not None and trades:
        mk = [(tr.book_side[0], tr.price, tr.quantity) for tr in trades if tr.taker_owner == 'market']
        if mk:
            events.append((t, 'trade', mk, None, s.mbb, s.mba, s.pos, s.ss_dir, s._in_window(t)))
    orig_obc(snap, trades)


s._fire, s.on_order_book_change, s._shock_trigger = fire, obc, trig
ensure_cache(ROOT / 'data/round4.csv')
res, curve = run_backtest(s, ROOT / 'data/round4.csv')
print(f'pnl {res.pnl:.0f} breach {res.breach_penalty:.0f}; fires {len(fires)}')
for i in a.ids:
    r = ev.iloc[i]
    print(f'\n=== shock {i}: jump {r.jump:+.0f}, onset {r.onset:.3f}, t_jump {r.t_jump:.3f}')
    for f in fires:
        if abs(f[0] - r.t_jump) <= 1.0:
            print(f'  FIRE {f[0] - r.t_jump:+.3f}s d {f[1]:+d} edge {f[2]:.1f} touch {f[3]}/{f[4]} pos {f[5]} ss {f[6]}->{f[7]} '
                  f'onset_est {f[8] - r.onset if f[8] is not None else None:+.3f} sg_t {f[9]}')
    for e in events:
        if abs(e[0] - r.t_jump) <= a.w:
            print(f'  {e[0] - r.t_jump:+.3f}s {e[1]:5s} {e[2]} {e[3] if e[3] is not None else ""} touch {e[4]}/{e[5]} pos {e[6]} ss {e[7]} win {e[8]}')
