"""Which readings feed the simple tracker, and how wrong are they? Runs r4_simple, records every f.quote call
(t, side, price, F before, accepted?) and joins the true S path and the per-insert class labels
(makers/cache/order_class.parquet: 0 base, 1 ladder, 2 ?, 3 r15, 4 TWAP). Prints the reading residual
(implied S - true S) by class, regime and distance from the mid, and how many base inserts were used.
uv run --with pyarrow python marek/research/round4/edge/tools/readings.py [--set k=v ...]"""
import argparse, ast, pathlib, sys
import numpy as np, pandas as pd
ROOT = pathlib.Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from fastsim.engine import load_strategy, run_backtest
from fastsim.tape import ensure_cache
R4 = pathlib.Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser(); ap.add_argument('--set', nargs='*', default=[])
ap.add_argument('--strat', default='strategies/marek/round4/r4_simple.py')
a = ap.parse_args()
over = {k: ast.literal_eval(v) for k, v in (kv.split('=', 1) for kv in a.set)}
s = load_strategy(a.strat, over)
import importlib.util
P = sys.modules[type(s).__module__].P
rec = []
orig_obc = s.on_order_book_change
state = {'f': None, 'mid': np.nan, 't_us': 0}


def hook_quote(f):
    oq = f.quote

    def quote(t, is_bid, price):
        m0, pss0, pv0, kv0 = f.mean, f.pss, f.pv, f.kv
        w = 1.0 if f.drought else pv0
        d = (1 - w) * P['d_c'] + w * P['d_v']
        y = price + P['y_half'] + d if is_bid else price - P['y_half'] - d
        oq(t, is_bid, price)
        rec.append((t, state['t_us'], 1 if is_bid else -1, price, y, m0, f.mean, pss0, pv0, kv0, state['mid'], int(f.n_rej == 0)))
    f.quote = quote


def obc(snap, trades):
    state['t_us'] = snap.timestamp_us
    if snap.bids and snap.asks:
        state['mid'] = 0.5 * (snap.bids[0].price + snap.asks[0].price)
    orig_obc(snap, trades)
    if state['f'] is None and s.f is not None:
        state['f'] = s.f
        hook_quote(s.f)


s.on_order_book_change = obc
ensure_cache(ROOT / 'data/round4.csv')
res, curve = run_backtest(s, ROOT / 'data/round4.csv')
print(f'pnl {res.pnl:.0f}; readings {len(rec)}')
r = pd.DataFrame(rec, columns=['t', 't_us', 'side', 'price', 'y', 'F0', 'F1', 'pss', 'pv', 'kv', 'mid', 'acc'])
sp = pd.read_parquet(R4 / 's_dynamics/s_path.parquet', columns=['t_us', 's_hat'])
st = sp.t_us.values; S = sp.s_hat.values
r['S'] = S[np.clip(np.searchsorted(st, r.t_us.values, 'right') - 1, 0, len(S) - 1)]
r['res'] = r.y - r.S
oc = pd.read_parquet(R4 / 'makers/cache/order_class.parquet')
r = r.merge(oc[['t_us', 'cls']], on='t_us', how='left')
r['cls'] = r.cls.fillna(-1).astype(int)
on = pd.read_csv(R4 / 'flow/shock_events.csv').onset.values
sh = np.zeros(len(r), bool)
for o in on:
    sh |= (r.t.values >= o - 5) & (r.t.values < o + 25)
r = r[~sh]
r['turb'] = r.pv > 0.5
r['dmid'] = (r.price - r.mid) * r.side  # negative = behind the touch side's mid
print('\nreadings by class (shock windows removed):')
print(r.groupby(['turb', 'cls']).agg(n=('res', 'size'), acc=('acc', 'mean'), res_mean=('res', 'mean'), res_sd=('res', 'std')).round(3).to_string())
print('\nbase readings in volatile by (price - mid) * side:')
b = r[(r.cls == 0) & r.turb]
for lo, hi in ((-99, -4), (-4, -2), (-2, -1), (-1, 0), (0, 99)):
    m = (b.dmid >= lo) & (b.dmid < hi)
    print(f'  dmid [{lo},{hi}): n {m.sum():7d} acc {b.acc[m].mean():.3f} res mean {b.res[m].mean():+.3f} sd {b.res[m].std():.3f} | F0-S mean {(b.F0 - b.S)[m].mean():+.3f}')
print('\nbase readings in volatile by (y - F0) (innovation):')
b = b.assign(inn=b.y - b.F0)
for lo, hi in ((-99, -3), (-3, -1), (-1, 1), (1, 3), (3, 99)):
    m = (b.inn >= lo) & (b.inn < hi)
    print(f'  inn [{lo},{hi}): n {m.sum():7d} acc {b.acc[m].mean():.3f} res mean {b.res[m].mean():+.3f} | true S - F0 mean {(b.S - b.F0)[m].mean():+.3f}')
n_base = ((oc.cls == 0).sum())
print(f'\nbase inserts on tape {n_base}, used as readings {int((r.cls == 0).sum())} (outside shock windows)')
