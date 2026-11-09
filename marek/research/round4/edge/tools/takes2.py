"""Real IOC takes with F at send: limit edge vs F, fills' edge vs F and markouts vs S, by regime and reducing/increasing;
also the calm takes' markout by the F-edge bucket (is a lower calm take threshold worth it?).
uv run --with pyarrow python marek/research/round4/edge/tools/takes2.py cache/r4_volsig_round4.npz"""
import sys, pathlib, numpy as np, pandas as pd
R4 = pathlib.Path(__file__).resolve().parents[2]
z = np.load(sys.argv[1]); od, fl, x = z['od'], z['fl'], z['x']
t, F, sd, pv, turb, pos, bb, ba, Y, kv, sgw = x.T
on = pd.read_csv(R4 / 'flow/shock_events.csv').onset.values
sp = pd.read_parquet(R4 / 's_dynamics/s_path.parquet', columns=['t_us', 's_hat'])
st = sp.t_us.values * 1e-6; S = sp.s_hat.values
def at(arr, tt, tq): return arr[np.clip(np.searchsorted(tt, tq, 'right') - 1, 0, len(arr) - 1)]
o = pd.DataFrame(od, columns=['t', 'side', 'price', 'qty', 'ioc', 'pos', 'rr', 'rc', 'turb', 'shock', 'ss', 'stale', 'F', 'sd'])
shk = np.zeros(len(o), bool)
for oo in on: shk |= (o.t.values >= oo - 5) & (o.t.values < oo + 25)
tk = o[(o.ioc == 1) & (o.ss == 0) & (o.shock == 0) & (o.pos.abs() <= 100) & ~shk].copy()
tk['edge'] = (tk.F - tk.price) * tk.side; tk['red'] = tk.side * tk.pos < 0
tk['S'] = at(S, st, tk.t.values); tk['edgeS'] = (tk.S - tk.price) * tk.side
print('takes: limit edge vs F (lower bound on fill edge)')
for (tb, rd), s in tk.groupby(['turb', 'red']):
    print(f'  turb={int(tb)} red={int(rd)}: n {len(s)} lots {int(s.qty.sum())} edge mean {np.average(s.edge, weights=s.qty):+.2f} '
          f'(F-S at send {np.average(s.F - s.S, weights=s.qty) * 0:+.2f}; edge vs S {np.average(s.edgeS, weights=s.qty):+.2f}) '
          f'sd mean {s.sd.mean():.2f}; q<=10 share {(s.qty <= 10).mean():.2f}')
f = pd.DataFrame(fl, columns=['t', 'side', 'price', 'qty', 'ioc', 'pos', 'F'])
f['turb'] = at(turb, t, f.t.values)
shk = np.zeros(len(f), bool)
for oo in on: shk |= (f.t.values >= oo - 5) & (f.t.values < oo + 25)
f = f[~shk & (f.pos.abs() <= 100)].copy()
f['red'] = f.side * f.pos < 0
f['eF'] = (f.F - f.price) * f.side
for h in (0, 1, 5, 20): f[f'mo{h}'] = (at(S, st, f.t.values + h) - f.price) * f.side
print('\nfills by F-edge bucket (calm IOC): markouts vs S')
c = f[(f.turb == 0) & (f.ioc == 1)]
for lo, hi in ((-9, 0), (0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1), (1, 1.5), (1.5, 9)):
    s = c[(c.eF >= lo) & (c.eF < hi)]
    if len(s) == 0: continue
    w = s.qty
    print(f'  eF [{lo:4},{hi:4}): n {len(s):5d} lots {int(w.sum()):7d} ' + ' '.join(f'mo{h} {np.average(s[f"mo{h}"], weights=w):+.2f}' for h in (0, 1, 5, 20)))
print('\nfills by F-edge bucket (calm passive): markouts vs S')
c = f[(f.turb == 0) & (f.ioc == 0)]
for lo, hi in ((-9, 0), (0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1), (1, 1.5), (1.5, 9)):
    s = c[(c.eF >= lo) & (c.eF < hi)]
    if len(s) == 0: continue
    w = s.qty
    print(f'  eF [{lo:4},{hi:4}): n {len(s):5d} lots {int(w.sum()):7d} red {np.average(s.red, weights=w):.2f} ' + ' '.join(f'mo{h} {np.average(s[f"mo{h}"], weights=w):+.2f}' for h in (0, 1, 5, 20)))
print('\nturb IOC fills by F-edge bucket')
c = f[(f.turb == 1) & (f.ioc == 1)]
for lo, hi in ((-9, 0), (0, 0.5), (0.5, 1), (1, 2), (2, 4), (4, 99)):
    s = c[(c.eF >= lo) & (c.eF < hi)]
    if len(s) == 0: continue
    w = s.qty
    print(f'  eF [{lo:4},{hi:4}): n {len(s):5d} lots {int(w.sum()):7d} red {np.average(s.red, weights=w):.2f} ' + ' '.join(f'mo{h} {np.average(s[f"mo{h}"], weights=w):+.2f}' for h in (0, 1, 5, 20)))
