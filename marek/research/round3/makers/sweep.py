"""Do sweeps skip levels while keeping the 2 ms clock? Count same-side maker pairs at exactly 2000*j us and their price step."""
import numpy as np, pandas as pd
o = pd.read_pickle('cache/chains.pkl')
m = o[~o.ioc.astype(bool)]
t = m.t_us.to_numpy(); sg = m.sg.to_numpy(); p = m.price.to_numpy()
pos = {tt: i for i, tt in enumerate(t)}
for j in range(1, 7):
    steps = []
    for i in range(len(t)):
        k = pos.get(t[i] + 2000*j)
        if k is not None and sg[k] == sg[i]:
            steps.append((p[k] - p[i]) * -sg[i])   # >0 = deeper
    s = pd.Series(steps)
    print(j, 'pairs', len(s), 'step dist', s.value_counts().head(6).to_dict())
# background: same for 2000*j + 1 us
for j in (1, 2, 3):
    c = sum(1 for i in range(len(t)) if (t[i] + 2000*j + 7) in pos)
    print('background offset +7us, j', j, c)
