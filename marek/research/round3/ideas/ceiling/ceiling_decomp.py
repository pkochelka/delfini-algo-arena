"""Ceiling angle, step 3: decompose a run's PnL by fill type and regime against the exact S.
uv run --with pyarrow python marek/research/round3/ideas/ceiling_decomp.py TRADES_CSV ORD_CSV EV_NPZ [PENALTY]
Fill edge = signed qty * (S(t_fill) - price), S = exact S after the last insert at or before the fill (a taker's
own kick included). Carry = sum pos * dS between inserts. PnL = edge + carry + final mark gap - penalty."""
import sys, pathlib, bisect
import numpy as np, pandas as pd
H = pathlib.Path(__file__).parent
sys.path.insert(0, str(H.parents[2] / 'strategy'))
import eval3
ROOT = H.parents[5]
on, reg = eval3.load_ctx('data/round3.csv')
z = np.load(ROOT / 'data/.cache/round3_S_oracle.npz')
TI, SA = z['t_us'], z['S']
trf, ordf, evf = sys.argv[1:4]
pen = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
tr = pd.read_csv(trf)
tr = tr[(tr.taker_owner == 'user') | (tr.maker_owner == 'user')].copy()
od = pd.read_csv(ordf)
LAB = {689: 'take', 701: 'take', 780: 'passive', 784: 'passive', 980: 'xunwind', 1004: 'xunwind', 1249: 'snipe',
       1298: 'drflat', 1304: 'drflat'}
od['typ'] = [LAB.get(int(l.split(':')[1]), l) for l in od.label]
typ = dict(zip(od.oid, od.typ))
tr['me_maker'] = tr.maker_owner == 'user'
tr['oid'] = np.where(tr.me_maker, tr.maker_order_id, tr.taker_order_id)
tr['typ'] = tr.oid.map(typ)
# our side: maker -> opposite of taker side; taker -> taker side
tr['sgn'] = np.where(tr.me_maker, np.where(tr.taker_side == 'B', -1, 1), np.where(tr.taker_side == 'B', 1, -1))
def S_at(t):
    j = np.searchsorted(TI, np.asarray(t), side='right') - 1
    return SA[np.maximum(j, 0)]
t = tr.timestamp_us.values
tr['S'] = S_at(t)
tr['edge'] = tr.sgn * tr.quantity * (tr.S - tr.price)
for h, nm in [(1e6, 'm1'), (10e6, 'm10'), (60e6, 'm60')]:
    tr[nm] = tr.sgn * tr.quantity * (S_at(t + h) - tr.price)
tr['reg'] = [eval3.label(x * 1e-6, on, reg) for x in t]
pd.set_option('display.width', 220)
g = tr.groupby(['reg', 'typ']).agg(fills=('quantity', 'size'), lots=('quantity', 'sum'), edge=('edge', 'sum'),
                                     m1=('m1', 'sum'), m10=('m10', 'sum'), m60=('m60', 'sum'))
g['edge_lot'] = g.edge / g.lots
g['m10_lot'] = g.m10 / g.lots
print(g.round(2))
print('totals by type:\n', tr.groupby('typ')[['quantity', 'edge', 'm1', 'm10', 'm60']].sum().round())
# carry
dq = tr.sgn * tr.quantity
pos_t = np.cumsum(dq.values)
fill_t = t
k = np.searchsorted(fill_t, TI, side='right') - 1  # fills at or before each insert
pos_at = np.where(k >= 0, pos_t[np.maximum(k, 0)], 0)
dS = np.diff(SA)
carry = pos_at[:-1] * dS  # pos held over (TI[i], TI[i+1]], with S jumping at TI[i+1]
labs = np.array([eval3.label(x * 1e-6, on, reg) for x in TI[1:]])
cr = {r: float(carry[labs == r].sum()) for r in ['calm', 'volatile', 'shock']}
print('carry by regime', {k_: round(v) for k_, v in cr.items()})
ed = tr.groupby('reg').edge.sum()
print('edge by regime', ed.round().to_dict())
mark = 9873.0
final_pos = pos_t[-1]
gap = final_pos * (mark - SA[-1])
print('final pos', final_pos, 'mark gap', round(gap), 'total edge+carry+gap-pen', round(tr.edge.sum() + carry.sum() + gap - pen))
# shock-window carry split: jump instant vs rest
sh_on = np.array(on)
# time pinned / inventory per regime from events
ev = np.load(evf)
et, ep = ev['t_us'], ev['pos']
dt = np.diff(et) * 1e-6
el = np.array([eval3.label(x * 1e-6, on, reg) for x in et[:-1:50]])
el = np.repeat(el, 50)[:len(dt)]
for r in ['calm', 'volatile', 'shock']:
    m = el == r
    w = dt[m]; p = np.abs(ep[:-1][m])
    print(r, 'time', round(w.sum()), 's  |pos|>=95 share', round((w * (p >= 95)).sum() / w.sum(), 3),
          ' >100 s', round((w * (p > 100)).sum(), 2), ' mean|pos|', round((w * p).sum() / w.sum(), 1))
tr.to_parquet(H / '_cache' / (pathlib.Path(ordf).stem + '_fills.parquet'))
