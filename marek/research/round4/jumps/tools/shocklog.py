"""Tick-level log around each shock jump (onset+2.0 .. onset+4.5): book, our state, sniper state -> cache/shock_<strat>.pkl"""
import argparse, ast, pathlib, sys, csv, pickle
ROOT = pathlib.Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from fastsim.engine import load_strategy, run_backtest
from fastsim.tape import ensure_cache
ap = argparse.ArgumentParser(); ap.add_argument('strat'); ap.add_argument('--set', nargs='*', default=[])
ap.add_argument('--tag', default='')
a = ap.parse_args()
over = {k: ast.literal_eval(v) for k, v in (kv.split('=', 1) for kv in a.set)}
R4 = ROOT / 'marek/research/round4'
on = [float(r['onset']) for r in csv.DictReader(open(R4 / 'flow/shock_events.csv'))]
s = load_strategy(a.strat, over)
log = {i: [] for i in range(len(on))}
orig = s.on_order_book_change
def hook(snap, trades):
    t = snap.timestamp_us * 1e-6
    k = None
    for i, o in enumerate(on):
        if o - 1.0 <= t <= o + 4.5: k = i; break
    pre = (s.sg_t, s.ss_dir, s.pos) if k is not None else None
    orig(snap, trades)
    if k is None: return
    bids = [(o.price, o.quantity, o.owner) for o in snap.bids[:25]]
    asks = [(o.price, o.quantity, o.owner) for o in snap.asks[:25]]
    tr = [(x.price, x.quantity, x.book_side, x.taker_owner) for x in trades]
    log[k].append(dict(t=t, sg=s.sg_t, onset=s.ss_onset, ss_dir=s.ss_dir, edge=s.ss_edge, pos=s.pos, F=s.f.mean if s.f else None,
                       bids=bids, asks=asks, trades=tr, nsn=s.n_snipes, orders=dict(s.orders)))
s.on_order_book_change = hook
ensure_cache(ROOT / 'data/round4.csv')
res, curve = run_backtest(s, ROOT / 'data/round4.csv')
print('pnl', res.pnl)
pickle.dump(dict(on=on, log=log), open(R4 / 'jumps/cache' / f'shock_{pathlib.Path(a.strat).stem}{a.tag}.pkl', 'wb'))
