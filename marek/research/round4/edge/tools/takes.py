"""IOC takes: how much stale quantity was left because the room was held by our own resting (non-IOC) orders on the
same side, by regime; and the F-edge of the lots left. Also fill markouts vs s_path by (regime, kind, reducing).
uv run --with pyarrow python marek/research/round4/edge/tools/takes.py cache/r4_volsig_round4.npz"""
import sys, pathlib, numpy as np, pandas as pd
R4 = pathlib.Path(__file__).resolve().parents[2]
z = np.load(sys.argv[1])
x, od, fl = z['x'], z['od'], z['fl']
t, F, sd, pv, turb, pos, bb, ba, Y, kv, sgw = x.T
sp = pd.read_parquet(R4 / 's_dynamics/s_path.parquet', columns=['t_us', 's_hat'])
st = sp.t_us.values * 1e-6; S = sp.s_hat.values
def at(arr, tt, tq):
    return arr[np.clip(np.searchsorted(tt, tq, 'right') - 1, 0, len(arr) - 1)]
o = pd.DataFrame(od, columns=['t', 'side', 'price', 'qty', 'ioc', 'pos', 'rr', 'rc', 'turb', 'shock', 'ss', 'stale'])
o['F'] = at(F, t, o.t.values)
o['S'] = at(S, st, o.t.values)
tk = o[(o.ioc == 1) & (o.ss == 0) & (o.shock == 0)].copy()
cap = 100
tk['room'] = cap - tk.side * tk.pos - tk.rr - tk.rc
tk['room_free'] = cap - tk.side * tk.pos - tk.rc
tk['want'] = np.minimum(tk.stale, tk.room_free)
tk['extra'] = np.maximum(tk.want - tk.qty, 0)
tk['edgeF'] = (tk.F - tk.price) * tk.side
tk['edgeS'] = (tk.S - tk.price) * tk.side
tk['red'] = (tk.side * tk.pos < 0)
print('IOC takes (non-snipe):', len(tk), 'lots', tk.qty.sum())
for nm, m in (('turb', tk.turb == 1), ('calm', tk.turb == 0)):
    s = tk[m]
    trunc = s[s.extra > 0]
    print(f'  {nm}: n {len(s)}, lots {s.qty.sum():.0f}, stale avail {s.stale.sum():.0f}, truncated takes {len(trunc)} '
          f'extra lots {s.extra.sum():.0f} (edgeF-weighted {(s.extra * s.edgeF).sum():.0f} ticks, edgeS-weighted {(s.extra * s.edgeS).sum():.0f}); '
          f'held by resting rr>0 in {(s.rr > 0).mean():.2f} of takes; mean edgeF {np.average(s.edgeF, weights=s.qty):.2f} edgeS {np.average(s.edgeS, weights=s.qty):.2f}')
    print(f'     stale > qty in {(s.stale > s.qty).mean():.2f} of takes; stale - qty total {(s.stale - s.qty).sum():.0f}; '
          f'room-limited (qty == room) share {(s.qty == s.room).mean():.2f}')
# fills: markouts vs S at 0, 1, 5, 20 s
f = pd.DataFrame(fl, columns=['t', 'side', 'price', 'qty', 'ioc', 'pos'])
f['turb'] = at(turb, t, f.t.values); f['shock'] = at(sgw, t, f.t.values)
f['red'] = f.side * f.pos < 0
for h in (0, 1, 5, 20):
    f[f'mo{h}'] = (at(S, st, f.t.values + h) - f.price) * f.side
f['F0'] = (at(F, t, f.t.values) - f.price) * f.side
print('\nfills: markout per lot vs S (ticks), lots-weighted; red = reduces |pos|')
for (tb, io, rd), s in f[f.shock == 0].groupby(['turb', 'ioc', 'red']):
    w = s.qty
    print(f'  turb={int(tb)} ioc={int(io)} red={int(rd)}: n {len(s):5d} lots {w.sum():7.0f}  F0 {np.average(s.F0, weights=w):+.2f}  '
          + '  '.join(f'mo{h} {np.average(s[f"mo{h}"], weights=w):+.2f}' for h in (0, 1, 5, 20))
          + f'  total mo5 {(s.mo5 * w).sum()/1e3:+.1f}k mo20 {(s.mo20 * w).sum()/1e3:+.1f}k')
