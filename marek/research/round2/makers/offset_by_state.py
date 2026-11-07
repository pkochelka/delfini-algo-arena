"""Offset D by flow-agent state (calm / vol / drought / post-jump) and the one-multiplier test:
D_state = m_state * (d0_calm + Exp(scale_calm))."""
import numpy as np, pandas as pd
from scipy import optimize
from offset import g, fit, prep, nll
import states

g = g.assign(st=states.state(g.t_us.to_numpy()))
rows = []
calm = None
for s_ in ['calm', 'vol', 'drought', 'post']:
    gg = g[g.st == s_]
    gg = gg.sample(min(len(gg), 80000), random_state=0)
    x, l, cs = fit(gg)
    xg, lg, _ = fit(gg, True)
    r = {'state': s_, 'n_all': int((g.st == s_).sum()), 'd0': x[0], 'scale': x[1], 'mean_D': x[0] + x[1], 'nll_free': l,
         'g_d0': xg[0], 'g_scale': xg[1], 'g_k': xg[2], 'clipped': cs, 'S_sd_med': gg.S_sd.median()}
    if s_ == 'calm':
        calm = x
    else:
        y, ylim, c = prep(gg)
        f = lambda m: nll([calm[0] * m, calm[1] * m], y, ylim, c, False) / len(gg)
        rm = optimize.minimize_scalar(f, bounds=(0.3, 6), method='bounded')
        r['m_one_mult'] = rm.x; r['nll_one_mult'] = rm.fun; r['dnll_per_obs'] = rm.fun - l
    rows.append(r)
df = pd.DataFrame(rows); pd.set_option('display.width', 250); print(df.round(4).to_string()); df.to_csv('offset_by_state.csv', index=False)
