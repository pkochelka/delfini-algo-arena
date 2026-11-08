import numpy as np, pandas as pd
for n in ['round3', 'round4']:
    d = pd.read_csv(f'data/{n}.csv')
    t = np.round(d.Time.values * 1e6).astype(np.int64)
    isT = ((d.Operation == 'Insert') & (d.Lifespan == 'F')).values
    isM = ((d.Operation == 'Insert') & (d.Lifespan == 'G')).values
    isC = (d.Operation == 'Cancel').values
    ti = np.where(isT)[0]
    # for each taker, events within next 5 ms by type and dt
    res = {'M': [], 'C': [], 'T': []}
    for i in ti:
        j = i + 1
        while j < len(t) and t[j] - t[i] <= 5000:
            k = 'M' if isM[j] else ('C' if isC[j] else 'T')
            res[k].append(t[j] - t[i]); j += 1
    print('=====', n, 'takers', len(ti))
    for k, v in res.items():
        v = np.array(v)
        h = np.bincount(v, minlength=5001)
        print(k, 'dt 0..10us', h[:11].tolist(), ' bins 0-100,100-500,500-1000,1-2ms,2-5ms per us:',
              [round(h[a:b].sum() / (b - a), 2) for a, b in [(1, 100), (100, 500), (500, 1000), (1000, 2000), (2000, 5001)]])
    # vol of makers at 1us after taker
    j = ti[(ti + 1 < len(t))]; j = j[(t[j + 1] - t[j] == 1) & isM[j + 1]]
    print('maker-at-1us vol counts:', d.Volume.iloc[j + 1].value_counts().head(8).to_dict(), 'times range', d.Time.iloc[j].min(), d.Time.iloc[j].max())
