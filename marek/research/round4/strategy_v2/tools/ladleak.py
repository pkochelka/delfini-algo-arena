"""Leaked ladder rungs (fed with lad on): gap to the previous rung of the chain, price step, depth behind touch."""
import sys, numpy as np, pandas as pd, pathlib
H = pathlib.Path(__file__).resolve().parents[1]; R4 = H.parent
cl = pd.read_parquet(R4 / 'makers/cache/order_class.parquet')
tape = pd.read_csv(H.parents[3] / 'data/round4.csv')
ins = tape[tape.Operation == 'Insert'].reset_index(drop=True)
cl['price'] = ins.Price.values; cl['side'] = ins.Side.values; cl['vol'] = ins.Volume.values
f = np.load(H / f'cache/fed_{sys.argv[1]}.npy'); fs = set(f[:, 0].astype(np.int64))
lad = cl[cl.cls == 1].copy(); lad['fed'] = lad.t_us.isin(fs)
lad['rung'] = lad.groupby('lad_chain').cumcount()
lad['dt'] = lad.groupby('lad_chain').t_us.diff(); lad['dp'] = lad.groupby('lad_chain').price.diff()
lk = lad[lad.fed & (lad.rung >= 1)]
print('leaked rung>=1:', len(lk)); print('dt', lk.dt.value_counts().head(8).to_dict()); print('dp', lk.dp.value_counts().head(6).to_dict())
lk0 = lad[lad.fed & (lad.rung == 0)]
# for leaked rung 0: was the next rung fed too / what is dt to next
nx = lad[lad.rung == 1].set_index('lad_chain')
j = lk0.join(nx[['t_us', 'fed', 'price']], on='lad_chain', rsuffix='_n')
print('leaked rung 0:', len(lk0), 'next rung fed share', j.fed_n.mean(), 'dt to next', (j.t_us_n - j.t_us).value_counts().head(5).to_dict())
# how many rows between consecutive rungs in the tape (other events)
allt = tape.Time.values
