import numpy as np, pandas as pd
r = {}
for n in ['round3', 'round4']:
    d = pd.read_csv(f'data/{n}.csv')
    r[n] = d
    mk = d[(d.Operation == 'Insert') & (d.Lifespan == 'G')]
    tk = d[(d.Operation == 'Insert') & (d.Lifespan == 'F')]
    vc = mk.Volume.value_counts()
    print(n, 'maker vol counts 1..30:', [int(vc.get(v, 0)) for v in range(1, 31)])
    tv = tk.Volume.value_counts()
    print(n, 'taker vol counts 1..30:', [int(tv.get(v, 0)) for v in range(1, 31)])
d = r['round4']
mk = d[(d.Operation == 'Insert') & (d.Lifespan == 'G')]
m15 = mk[mk.Volume == 15]
print('r4 maker vol15 n', len(m15), 'side', m15.Side.value_counts().to_dict())
print('vol15 per 100s:', np.bincount(np.floor(m15.Time.values / 100).astype(int)).tolist())
