"""Per-taker kick: sign*(S after - S before) on a path TAG, by filled bin: mean, sd, sd minus posterior noise. Is xi additive
(sd flat in f) or multiplicative (sd ~ f^e)? Uses the np path (no impact prior) by default; RUN with TAPE for r2."""
import sys, numpy as np, pandas as pd
import prep
tag = sys.argv[1] if len(sys.argv) > 1 else 'np'
d = prep.build(__import__('os').environ.get('TAPE', 'data/round3.csv')); r = dict(np.load(prep.common.CACHE / f'smooth_{tag}.npz'))
m = r['mean']; sd = r['sd']; vol = r['vol'].astype(bool); sh = r['shock'].astype(bool)
tk = np.where(d['ioc'] & (d['filled'] > 0))[0]; tk = tk[tk < len(m) - 1]
y = d['sign'][tk] * (m[tk + 1] - m[tk]); f = d['filled'][tk]
nv = sd[tk] ** 2 + sd[tk + 1] ** 2   # upper bound on noise from posterior (ignores covariance)
bins = [0, 5, 20, 50, 100, 200, 400, 800, 1e9]
for nm, reg in (('calm', ~vol & ~sh), ('vol', vol & ~sh)):
    s = reg[tk] & reg[tk + 1]
    df = pd.DataFrame(dict(f=f[s], y=y[s], nv=nv[s]))
    g = df.groupby(pd.cut(df.f, bins)).agg(n=('y', 'size'), f=('f', 'mean'), y=('y', 'mean'), sd=('y', 'std'), nv=('nv', 'mean'))
    g['ratio'] = g.y / g.f ** 0.61
    g['sd_rel'] = g.sd / g.y
    print('==', nm); print(g.round(4).to_string())
