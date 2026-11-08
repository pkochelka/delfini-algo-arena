"""Log r4_best-family filter state every 10 ms (t, F, sd, pv, turb, pos, bb, ba, kv, kPv, Y, exc, sg) -> npz.
    uv run python strategies/peter/r4_vlog.py STRAT.py OUT.npz [--tape data/round4.csv] [--set '{"k":v}']"""
import argparse, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fastsim.engine import load_strategy, run_backtest
ap = argparse.ArgumentParser(); ap.add_argument('strat'); ap.add_argument('out')
ap.add_argument('--tape', default='data/round4.csv'); ap.add_argument('--set', default='{}')
a = ap.parse_args()
s = load_strategy(a.strat, json.loads(a.set) or None)
rows = []; nxt = [0.0]
orig = s.on_order_book_change
def hook(snap, trades):
    orig(snap, trades)
    t = snap.timestamp_us * 1e-6
    f = s.f
    if t >= nxt[0] and f is not None and snap.bids and snap.asks:
        nxt[0] = t + 0.01
        rows.append((t, f.mean, f.sd, f.pv, float(getattr(s, 'turb', 0)), s.pos, snap.bids[0].price, snap.asks[0].price,
                     f.kv, f.kPv, getattr(f, 'Y', 0.0), f.exc, float(s.sg_t is not None)))
s.on_order_book_change = hook
res, curve = run_backtest(s, ROOT / a.tape)
np.savez(a.out, x=np.array(rows))
print('pnl', res.pnl, a.out)
