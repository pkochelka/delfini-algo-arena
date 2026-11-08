"""Pre-shock predictors of the jump sign, pooled over round 2 and round 3 (exact S paths).
usage: uv run --with pyarrow python jump_tell.py
"""
import numpy as np, pandas as pd
def load(rnd):
    S = pd.read_parquet(f'marek/research/{rnd}/s_dynamics/s_path.parquet', columns=['t_us', 's_hat'])
    return S.t_us.values, S.s_hat.values
out = []
for rnd, shf in (('round3', 'marek/research/round3/s_dynamics/shocks.csv'), ('round2', None)):
    st, sh = load(rnd)
    s_at = lambda t: sh[np.clip(np.searchsorted(st, t, side='right') - 1, 0, None)]
    if shf:
        d = pd.read_csv(shf); on = d.onset.values * 1e6; J = d.jump.values
    else:
        d = pd.read_csv('marek/research/round2/flow/shock_events.csv'); on = d.onset.values * 1e6
        J = s_at(on + 3.3e6) - s_at(on + 2.9e6)
    s0 = s_at(on)
    feats = {'S-10000': s0 - 10000, 'S-S(start)': s0 - sh[0]}
    for w in (10, 30, 60, 120, 300):
        feats[f'pre{w}'] = s0 - s_at(on - w * 1e6)
    feats['S-prevjumpS'] = s0 - np.r_[sh[0], s_at(on[:-1] + 3.5e6)]
    feats['prevJ'] = np.r_[0, J[:-1]]
    for k, v in feats.items():
        out.append((rnd, k, np.corrcoef(v, J)[0, 1]))
    print(rnd, 'n', len(J), 'J sd', J.std().round(1))
df = pd.DataFrame(out, columns=['rnd', 'feat', 'corr']).pivot(index='feat', columns='rnd', values='corr')
print(df.round(3))
