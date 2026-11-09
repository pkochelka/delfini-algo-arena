"""Markouts of our fills (fills.py npy) against s_path s_hat, by regime / kind / joiner relation / S - mid bucket.
uv run --with pyarrow python mk.py FILLS.npy"""
import sys, numpy as np, pandas as pd, pathlib
R4 = pathlib.Path(__file__).resolve().parents[2]
z = np.load(sys.argv[1])
tu, sg, q, px, kd, F, sd, pv, turb, pos, bb, ba, js, q0, age = z.T
sp = pd.read_parquet(R4 / 's_dynamics/s_path.parquet', columns=['t_us', 's_hat'])
st = sp.t_us.values; sh = sp.s_hat.values
def S_at(u): return sh[np.clip(np.searchsorted(st, u, side='right') - 1, 0, len(sh) - 1)]
reg = pd.read_csv(R4 / 'flow/regimes_1s.csv'); shk = pd.read_csv(R4 / 'flow/shock_events.csv')
t = tu * 1e-6
vol = np.zeros(len(t), bool)
for a, b, g in reg[['start', 'end', 'regime']].values:
    if g == 'volatile': vol |= (t >= a) & (t < b)
sh_ = np.zeros(len(t), bool)
for o in shk.onset.values: sh_ |= (t >= o - 1) & (t < o + 20)
S0 = S_at(tu); mid = 0.5 * (bb + ba)
H = (0, 0.1, 0.5, 1, 2, 5, 10)
M = {h: sg * (S_at(tu + h * 1e6) - px) for h in H}
def row(nm, m):
    if m.sum() == 0: return
    w = q[m]
    print(f'{nm:34s} fills {m.sum():6d} lots {w.sum():8.0f} ' + ' '.join(f'{h:>4}s {np.average(M[h][m], weights=w):+6.2f}' for h in H)
          + f'  pnl@5s {np.sum(M[5][m] * w)/1e3:+7.1f}k')
calm = ~vol & ~sh_
for nm, m in [('calm', calm), ('vol', vol & ~sh_), ('shock', sh_)]:
    row(nm, m); row(nm + ' passive', m & (kd == 0)); row(nm + ' ioc', m & (kd == 1))
print('--- calm passive by joiner relation')
cp = calm & (kd == 0)
row('with agent (our side = agent side)', cp & (js == sg)); row('against agent', cp & (js == -sg)); row('no agent', cp & (js == 0))
print('--- calm passive by z = side * (S - mid) at fill (S from s_path)')
zz = sg * (S0 - mid)
for lo, hi in [(-9, -1), (-1, -0.5), (-0.5, -0.25), (-0.25, 0), (0, 0.25), (0.25, 0.5), (0.5, 9)]:
    row(f'z in [{lo},{hi})', cp & (zz >= lo) & (zz < hi))
print('--- calm passive by z_F = side * (F - mid) at fill (our filter)')
zf = sg * (F - mid)
for lo, hi in [(-9, -1), (-1, -0.5), (-0.5, -0.25), (-0.25, 0), (0, 0.25), (0.25, 0.5), (0.5, 9)]:
    row(f'zF in [{lo},{hi})', cp & (zf >= lo) & (zf < hi))
print('--- calm passive by edge at fill side*(F - px)')
e = sg * (F - px)
for lo, hi in [(-9, 0), (0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1), (1, 9)]:
    row(f'edgeF in [{lo},{hi})', cp & (e >= lo) & (e < hi))
print('--- calm passive by position before (signed by fill side: + = adds to position)')
ps = sg * pos
for lo, hi in [(-101, -50), (-50, 0), (0, 1), (1, 50), (50, 101)]:
    row(f'pos*side in [{lo},{hi})', cp & (ps >= lo) & (ps < hi))
print('--- calm passive by fill size')
for lo, hi in [(1, 5), (5, 20), (20, 60), (60, 101)]:
    row(f'qty in [{lo},{hi})', cp & (q >= lo) & (q < hi))
print('--- calm passive by queue ahead at post / order age')
for lo, hi in [(0, 1), (1, 50), (50, 200), (200, 1e9)]:
    row(f'q ahead [{lo},{hi})', cp & (q0 >= lo) & (q0 < hi))
for lo, hi in [(0, 0.05), (0.05, 0.2), (0.2, 1), (1, 1e9)]:
    row(f'age [{lo},{hi})', cp & (age >= lo) & (age < hi))
