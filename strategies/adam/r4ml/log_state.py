"""Log r4_best's filter state plus raw book/flow features every 20 ms of a run -> npz (ML feasibility dataset).

    .venv/Scripts/python strategies/adam/r4ml/log_state.py [--strat strategies/marek/round4/r4_best.py]
        [--tape data/round4_public.csv] [--out logs/r4ml/state_r4.npz] [--dt 0.02]

Columns (COLS below): filter state (F, sd, P(volatile), Kalman velocity, transient Y, turbulent flag, our position),
touch prices, sizes at the first 3 levels per side, cumulative signed / unsigned market-taker volume and count,
cumulative market inserts per side, and the running PnL is in `curve`.
"""
import argparse
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from fastsim.engine import load_strategy, run_backtest  # noqa: E402
from fastsim.tape import ensure_cache  # noqa: E402

COLS = ['t', 'F', 'sd', 'pv', 'kv', 'Y', 'turb', 'pos', 'bb', 'ba',
        'bq1', 'bq2', 'bq3', 'aq1', 'aq2', 'aq3', 'nb', 'na',
        'cum_sv', 'cum_v', 'cum_n', 'cum_ins_b', 'cum_ins_a', 'cum_big']

ap = argparse.ArgumentParser()
ap.add_argument('--strat', default='strategies/marek/round4/r4_best.py')
ap.add_argument('--tape', default='data/round4_public.csv')
ap.add_argument('--out', default='logs/r4ml/state_r4.npz')
ap.add_argument('--dt', type=float, default=0.02)
a = ap.parse_args()

s = load_strategy(str(ROOT / a.strat), {})
rows = []
st = {'nxt': 0.0, 'sv': 0.0, 'v': 0.0, 'n': 0, 'ib': 0, 'ia': 0, 'big': 0, 'seen': set(), 'prev_b': {}, 'prev_a': {}}
orig = s.on_order_book_change


def lvl(side, i):
    return side[i].quantity if len(side) > i else 0


def hook(snap, trades):
    orig(snap, trades)
    for tr in trades:
        if tr.taker_owner == 'market':
            sg = 1 if tr.book_side == 'ask' else -1
            st['sv'] += sg * tr.quantity
            st['v'] += tr.quantity
            if tr.taker_order_id not in st['seen']:
                st['seen'].add(tr.taker_order_id)
                st['n'] += 1
            if tr.quantity >= 100:
                st['big'] += sg
    bids, asks = snap.bids, snap.asks
    # crude insert counter: touch quantity growth at an unchanged touch price
    if bids:
        pb = st['prev_b']
        if pb.get('p') == bids[0].price and bids[0].quantity > pb.get('q', 0):
            st['ib'] += 1
        st['prev_b'] = {'p': bids[0].price, 'q': bids[0].quantity}
    if asks:
        pa = st['prev_a']
        if pa.get('p') == asks[0].price and asks[0].quantity > pa.get('q', 0):
            st['ia'] += 1
        st['prev_a'] = {'p': asks[0].price, 'q': asks[0].quantity}
    t = snap.timestamp_us * 1e-6
    f = s.f
    if t >= st['nxt'] and f is not None and bids and asks:
        st['nxt'] = t + a.dt
        rows.append((t, f.mean, f.sd, f.pv, f.kv, f.Y, float(s.turb), s.pos, bids[0].price, asks[0].price,
                     lvl(bids, 0), lvl(bids, 1), lvl(bids, 2), lvl(asks, 0), lvl(asks, 1), lvl(asks, 2),
                     len(bids), len(asks), st['sv'], st['v'], st['n'], st['ib'], st['ia'], st['big']))


s.on_order_book_change = hook

# ---- fills: (t, side +1 buy / -1 sell, qty, price, ioc, pos_before, order age s, order price - touch same side,
#              F, sd, pv, kv, Y, turb, bb, ba, bq1, aq1)
FCOLS = ['t', 'side', 'qty', 'price', 'ioc', 'pos0', 'age', 'F', 'sd', 'pv', 'kv', 'Y', 'turb', 'bb', 'ba', 'bq1', 'aq1']
fills = []
meta = {}  # order id -> (ioc, send time)
book = {'bb': 0, 'ba': 0, 'bq1': 0, 'aq1': 0}
orig_send = s.send_order


def send_hook(req):
    oid = orig_send(req)
    meta[oid] = (float(req.order_type.name == 'IOC'), s.now_us() * 1e-6)
    return oid


s.send_order = send_hook
orig_pou = s.on_private_order_update


def pou_hook(ev):
    if ev.filled_quantity and ev.status.name in ('PARTIALLY_FILLED', 'FILLED'):
        o = s.orders.get(ev.order_id)
        ioc, t0 = meta.get(ev.order_id, (float('nan'), float('nan')))
        f = s.f
        side = 1 if (o is not None and o[0].name == 'BUY') else -1
        if o is not None and f is not None:
            t = ev.timestamp_us * 1e-6
            fills.append((t, side, ev.filled_quantity, ev.fill_price, ioc, s.pos, t - t0, f.mean, f.sd, f.pv, f.kv, f.Y,
                          float(s.turb), book['bb'], book['ba'], book['bq1'], book['aq1']))
    orig_pou(ev)


s.on_private_order_update = pou_hook
orig2 = s.on_order_book_change


def book_hook(snap, trades):
    if snap.bids and snap.asks:
        book.update(bb=snap.bids[0].price, ba=snap.asks[0].price, bq1=snap.bids[0].quantity, aq1=snap.asks[0].quantity)
    orig2(snap, trades)


s.on_order_book_change = book_hook
ensure_cache(ROOT / a.tape)
res, curve = run_backtest(s, ROOT / a.tape)
print('pnl', res.pnl, 'rows', len(rows))
out = ROOT / a.out
out.parent.mkdir(parents=True, exist_ok=True)
np.savez(out, x=np.array(rows, dtype=float), cols=np.array(COLS), curve=np.array(curve),
         fills=np.array(fills, dtype=float), fcols=np.array(FCOLS))
print('fills', len(fills))
print(out)
