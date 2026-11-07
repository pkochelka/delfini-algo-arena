"""Rolling 60 s fits of the offset scale vs realized S volatility in the same window."""
import numpy as np, pandas as pd
from offset import g, o, fit

t = o.t_us.to_numpy(); S = o.S.to_numpy()
grid = np.arange(0, 6000e6 + 1, 100_000)
Sg = S[np.clip(np.searchsorted(t, grid, 'right') - 1, 0, None)]
d = np.diff(Sg)  # 100 ms increments
rows = []
W = 60
for w0 in range(0, 6000, W):
    m = (g.t_us >= w0 * 1e6) & (g.t_us < (w0 + W) * 1e6)
    gg = g[m]
    if len(gg) > 6000:
        gg = gg.sample(6000, random_state=0)
    x, l, cs = fit(gg)
    dd = d[w0 * 10:(w0 + W) * 10]
    rows.append({'t0': w0, 'd0': x[0], 'scale': x[1], 'clipped': cs,
                 'rv100ms': np.sqrt(np.mean(dd ** 2) * 10), 'rv_robust': 1.4826 * np.median(np.abs(dd)) * np.sqrt(10),
                 'n_jump1': int((np.abs(dd) > 1.5).sum()), 'S_sd_med': float(np.median(o.S_sd[(o.t_us >= w0 * 1e6) & (o.t_us < (w0 + W) * 1e6)]))})
df = pd.DataFrame(rows)
df.to_csv('offset_rolling60.csv', index=False)
print(df.round(3).to_string())
print(df[['scale', 'd0', 'rv100ms', 'rv_robust', 'n_jump1', 'S_sd_med']].corr().round(2))
