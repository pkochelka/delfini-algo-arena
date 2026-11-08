import sys
import numpy as np, pandas as pd
for path in sys.argv[1:]:
    d = pd.read_csv(path)
    print('=====', path)
    t = np.round(d.Time.values * 1e6).astype(np.int64)
    dt = np.diff(t)
    idx = np.where(dt == 1)[0]
    a = d.iloc[idx]; b = d.iloc[idx + 1]
    key = (a.Operation.str[0] + a.Lifespan.fillna('-').values + '>' + b.Operation.str[0].values + b.Lifespan.fillna('-').values)
    print('1us pairs:', pd.Series(key.values).value_counts().head(8).to_dict())
    same = (a.OrderId.values == b.OrderId.values).mean()
    print('1us same id', same)
    # dt=0
    print('dt==0 count', (dt == 0).sum())
    # mid proxy: maker insert price mean per 10s
    mk = d[(d.Operation == 'Insert') & (d.Lifespan == 'G')]
    tk = d[(d.Operation == 'Insert') & (d.Lifespan == 'F')]
    g = mk.groupby(np.floor(mk.Time / 60)).Price.median()
    print('per-min median maker price, every 10 min:', g.iloc[::10].round(0).tolist())
    dd = g.diff().dropna()
    print('1-min drift sd', dd.std().round(2), 'mean', dd.mean().round(3), 'lag1 acf', dd.autocorr().round(3))
    # taker signed volume by minute vs price change
    sg = np.where(tk.Side == 'B', 1, -1) * tk.Volume
    sv = pd.Series(sg.values).groupby(np.floor(tk.Time.values / 60)).sum()
    print('corr signed taker vol/min with price change/min', np.corrcoef(sv.loc[dd.index], dd)[0, 1].round(3))
    print('taker side frac by 10 min (B):', tk.groupby(np.floor(tk.Time / 600)).Side.apply(lambda s: round((s == 'B').mean(), 3)).tolist())
    print('taker rate per 10min', tk.groupby(np.floor(tk.Time / 600)).size().tolist())
