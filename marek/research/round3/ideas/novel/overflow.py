"""'Overflow' quoting: a resting order at the BACK of level P is filled only by a taker that empties every market order
up to P, and only with what that taker has left. In the replay the S path already carries the taker's original
impact (c * filled^e of its ORIGINAL fill), so this extra volume is impact-free at the margin.
Virtual order at P = round(S_pre + side*k) + side*o (k = 1 calm, 2.55 volatile), size Q, on every taker, no room limit.
Markouts on exact S. usage: uv run --with pyarrow python overflow.py NULL_LOGDIR
"""
import sys, csv, numpy as np, pandas as pd
nd = sys.argv[1]
S = pd.read_parquet('marek/research/round3/s_dynamics/s_path.parquet', columns=['t_us', 's_hat', 's_filt', 'vol'])
st = S.t_us.values; sh = S.s_hat.values; vol = S.vol.values
import os
PLACE = (S.s_filt.values if os.environ.get('CAUSAL') else sh)  # placement from the causal filter (CAUSAL=1) or the smoother
tape = pd.read_csv('data/round3.csv', usecols=['Time', 'Operation', 'OrderId', 'Side', 'Volume', 'Price', 'Lifespan'])
ins = tape[tape.Operation == 'Insert'].reset_index(drop=True)
ti = np.where(ins.Lifespan.values == 'F')[0]
tk = pd.DataFrame(dict(oid=ins.OrderId.values[ti], ts=st[ti], side=np.where(ins.Side.values[ti] == 'B', 1, -1),
                       q=ins.Volume.values[ti], L=ins.Price.values[ti], s0=PLACE[ti - 1], v=vol[ti]))
cols = ['ts', 'price', 'tside', 'qq', 'tid', 'mid', 'town', 'mown']
tr = pd.read_csv(f'{nd}/trades.csv', names=cols, header=0)
tr = tr[tr.town == 'market']
tr = tr[tr.tid.isin(set(tk.oid))]
on = np.array([float(r['onset']) for r in csv.DictReader(open('marek/research/round3/flow/shock_events.csv'))]) * 1e6
bad = np.zeros(len(tk), bool)
for o in on: bad |= (tk.ts.values >= o - 1e6) & (tk.ts.values < o + 20e6)
tk['bad'] = bad
def s_at(x):
    return sh[np.clip(np.searchsorted(st, x, side='right') - 1, 0, None)]
g = tr.groupby('tid')
fills = {k: (v.price.values, v.qq.values) for k, v in g}
for Q in (100, 30):
    for o in (-1, 0, 1):
        res = {}
        for vv in (False, True):
            m = (tk.v.values == vv) & ~tk.bad.values
            sub = tk[m]
            k = 2.55 if vv else 1.0
            P = np.round(sub.s0.values + sub.side.values * k).astype(int) + sub.side.values * o
            ours = np.zeros(len(sub))
            for j, (oid, sd, q, p) in enumerate(zip(sub.oid.values, sub.side.values, sub.q.values, P)):
                pr, qq = fills.get(oid, (np.array([]), np.array([])))
                le = qq[(pr <= p) if sd > 0 else (pr >= p)].sum()
                lim_ok = (sub.L.values[j] >= p) if sd > 0 else (sub.L.values[j] <= p)
                ours[j] = min(Q, max(0, q - le)) if lim_ok else 0
            row = {'lots': ours.sum(), 'n': (ours > 0).sum()}
            for h in (0.2, 1, 10, 30):
                row[f'e{h}'] = (ours * sub.side.values * (P - s_at(sub.ts.values + h * 1e6))).sum()
            res['vol' if vv else 'calm'] = row
        print(f'Q={Q} o={o:+d}', {r: {kk: round(vv2) for kk, vv2 in d.items()} for r, d in res.items()})

# ---- conditional on a run's inventory (room on the side, reducing vs adding): python overflow.py NULL RUN
if len(sys.argv) > 2:
    rt = pd.read_csv(f'{sys.argv[2]}/trades.csv', names=cols, header=0)
    u = rt[(rt.town == 'user') | (rt.mown == 'user')]
    sgn = np.where(((u.town == 'user') & (u.tside == 'B')) | ((u.mown == 'user') & (u.tside == 'A')), 1, -1) * u.qq.values
    ut = u.ts.values; cum = np.cumsum(sgn)
    for VV, KK in ((True, 2.55), (False, 1.0)):
      m = (tk.v.values == VV) & ~tk.bad.values
      sub = tk[m]
      j = np.searchsorted(ut, sub.ts.values, side='left') - 1
      pos = np.where(j >= 0, cum[np.clip(j, 0, None)], 0)
      P = np.round(sub.s0.values + sub.side.values * KK).astype(int)
      ours = np.zeros(len(sub))
      for jj, (oid, sd, q, p_) in enumerate(zip(sub.oid.values, sub.side.values, sub.q.values, P)):
          pr, qq = fills.get(oid, (np.array([]), np.array([])))
          le = qq[(pr <= p_) if sd > 0 else (pr >= p_)].sum()
          ours[jj] = min(100, max(0, q - le)) if ((sub.L.values[jj] >= p_) if sd > 0 else (sub.L.values[jj] <= p_)) else 0
      # we sell to buy takers (side +1): reducing if pos > 0
      reduce = np.sign(pos) == sub.side.values
      pinned_block = (sub.side.values > 0) & (pos <= -100) | (sub.side.values < 0) & (pos >= 100)
      e1 = ours * sub.side.values * (P - s_at(sub.ts.values + 1e6)); e30 = ours * sub.side.values * (P - s_at(sub.ts.values + 30e6))
      for nm, mm in (('reducing', reduce & ~pinned_block), ('adding (room)', ~reduce & ~pinned_block), ('blocked', pinned_block)):
          print(f'{"vol" if VV else "calm"} overflow Q=100 o=0 {nm}: lots {ours[mm].sum():.0f} e1 {e1[mm].sum():.0f} e30 {e30[mm].sum():.0f}')
