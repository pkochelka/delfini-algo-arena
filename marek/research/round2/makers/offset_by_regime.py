"""Offset fit by volatility regime (rv3 bins), exponential and gamma shapes."""
import numpy as np, pandas as pd
from offset import g, fit
from regime import regime_at

rv, hot = regime_at(g.t_us.to_numpy())
g = g.assign(rv=rv)
rows = []
for lo, hi in [(0, 0.4), (0.4, 0.55), (0.55, 0.7), (0.7, 1.1), (1.1, 1.6), (1.6, 2.2), (2.2, 3), (3, 99)]:
    gg = g[(g.rv >= lo) & (g.rv < hi)]
    gg = gg.sample(min(len(gg), 60000), random_state=0)
    x, l, cs = fit(gg)
    xg, lg, _ = fit(gg, True)
    rows.append({'rv_bin': f'{lo}-{hi}', 'n_all': int(((g.rv >= lo) & (g.rv < hi)).sum()), 'rv_mean': gg.rv.mean(),
                 'd0': x[0], 'scale': x[1], 'mean_D': x[0] + x[1], 'g_d0': xg[0], 'g_scale': xg[1], 'g_k': xg[2],
                 'dLL_gamma_per_obs': l - lg, 'clipped': cs, 'S_sd_med': gg.S_sd.median()})
df = pd.DataFrame(rows); print(df.round(4).to_string()); df.to_csv('offset_by_regime.csv', index=False)
