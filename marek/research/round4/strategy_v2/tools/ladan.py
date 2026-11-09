"""Class breakdown of the inserts fed to the filter (ladchk.py output) vs order_class labels and vs lad_on=0."""
import numpy as np, pandas as pd, pathlib
H = pathlib.Path(__file__).resolve().parents[1]; R4 = H.parent
cl = pd.read_parquet(R4 / 'makers/cache/order_class.parquet')
sp = pd.read_parquet(R4 / 's_dynamics/s_path.parquet', columns=['vol', 'shock'])
tape = pd.read_csv(H.parents[3] / 'data/round4.csv', usecols=['Operation', 'Lifespan', 'Volume', 'Price', 'Side'])
ins = tape[tape.Operation == 'Insert'].reset_index(drop=True)
assert len(ins) == len(cl), (len(ins), len(cl))
cl['vol'] = sp.vol.values; cl['shock'] = sp.shock.values
mk = cl[cl.cls.isin([0, 1, 2])].copy()
import sys; f1 = np.load(H / f'cache/fed_{sys.argv[1] if len(sys.argv) > 1 else "lad1"}.npy'); f0 = np.load(H / 'cache/fed_lad0.npy')
s1 = set(f1[:, 0].astype(np.int64)); s0 = set(f0[:, 0].astype(np.int64))
mk['fed1'] = mk.t_us.isin(s1); mk['fed0'] = mk.t_us.isin(s0)
print('maker inserts', len(mk), 'unique t', mk.t_us.nunique())
for reg, m in [('calm', ~mk.vol & ~mk.shock), ('vol', mk.vol & ~mk.shock), ('shock', mk.shock)]:
    for c in (0, 1, 2):
        x = mk[m & (mk.cls == c)]
        print(f'{reg:5s} cls {c}: n {len(x):7d}  seen&fed(lad off) {x.fed0.mean():.3f}  fed(lad on) {x.fed1.mean():.3f}  '
              f'dropped by ladder filter {(x.fed0 & ~x.fed1).mean():.4f}  leak(ladder fed) {x.fed1.mean() if c == 1 else float("nan"):.3f}')
# ladder rungs that leak: by rung index within chain and by chain length
lad = mk[mk.cls == 1].copy()
lad['rung'] = lad.groupby('lad_chain').cumcount(); lad['L'] = lad.groupby('lad_chain').t_us.transform('size')
print('ladder leak by rung index (fed with lad on):')
print(lad.groupby(lad.rung.clip(upper=5)).fed1.agg(['size', 'mean']).T.round(3))
print('ladder leak by chain length, rung 0 only:')
r0 = lad[lad.rung == 0]
print(r0.groupby(r0.L.clip(upper=6)).fed1.agg(['size', 'mean']).T.round(3))
print('ladder leak rung 0, by vol:'); print(r0.groupby('vol').fed1.agg(['size', 'mean']).T.round(3))
