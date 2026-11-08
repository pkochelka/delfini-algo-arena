"""Post-jump takers: is the jump-side head explained by the side law Phi((S-mid)/4.42)? Observed vs predicted jump-side
share and counts per window after t_jump; burst count vs |J| and regime."""
import numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import norm
D = Path(__file__).parent; C = D / 'cache'
tk = pd.read_parquet(C / 'takers_s.parquet'); X = pd.read_csv(D / 'shock_events.csv')
t = tk.t.to_numpy(); s = tk.sg.to_numpy(); x = (tk.s_hat - tk.mid).to_numpy(); p = norm.cdf(x / 4.42); nt = ~tk.tw.to_numpy()
rows = []
for _, e in X.iterrows():
    sj = np.sign(e.jump)
    for a, b in [(-1, 0), (0, .05), (.05, .1), (.1, .2), (.2, .5), (.5, 1), (1, 2), (2, 3.5), (3.5, 8)]:
        m = nt & (t >= e.t_jump + a) & (t < e.t_jump + b) & np.isfinite(x)
        pj = np.where(sj > 0, p[m], 1 - p[m])
        rows.append(dict(a=a, b=b, n=m.sum(), obs=(s[m] == sj).sum(), pred=pj.sum(), absx=np.abs(x[m]).sum()))
R = pd.DataFrame(rows).groupby(['a', 'b']).sum()
R['obs_share'] = R.obs / R.n; R['pred_share'] = R.pred / R.n; R['per_shock_rate'] = R.n / 28 / (R.index.get_level_values(1) - R.index.get_level_values(0)); R['mean|S-mid|'] = R.absx / R.n
print(R.round(3))
base = np.where(X.regime_at_jump == 'volatile', 12.3, 8.2) * 3.5; ex = X.burst_takers_3p5s - base
for g in ['calm', 'volatile']:
    m = X.regime_at_jump == g; print(g, 'extra burst mean %.1f, corr with |J| %.2f' % (ex[m].mean(), np.corrcoef(ex[m], np.abs(X.jump[m]))[0, 1]))
b = np.polyfit(np.abs(X.jump), ex, 1); print('extra = %.1f + %.3f*|J|' % (b[1], b[0]))
