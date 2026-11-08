"""Nonparametric impact curve: smoother run with zero impact prior and wide per-taker variance (tag np),
mean sign*dS across each taker by filled-size bin, per regime."""
import numpy as np, pandas as pd, sys
import prep
tag = sys.argv[1] if len(sys.argv) > 1 else 'np'
d = prep.build(); r = dict(np.load(prep.common.CACHE / f'smooth_{tag}.npz'))
m = r['mean']; vol = r['vol'].astype(bool); sh = r['shock'].astype(bool)
tk = np.where(d['ioc'])[0]; tk = tk[tk < len(m) - 1]
dS = d['sign'][tk] * (m[tk + 1] - m[tk])
f = d['filled'][tk]; v = d['volume'][tk]
bins = [-1, 0, 2, 5, 10, 20, 40, 80, 160, 320, 640, 1280, 1e9]
for nm, reg in [('calm', ~vol & ~sh), ('vol', vol & ~sh)]:
    sel = reg[tk] & reg[tk + 1]
    df = pd.DataFrame(dict(f=f[sel], v=v[sel], y=dS[sel]))
    g = df.groupby(pd.cut(df.f, bins)).agg(n=('y', 'size'), f=('f', 'mean'), y=('y', 'mean'), sd=('y', 'std'))
    g['se'] = g.sd / np.sqrt(g.n)
    g['y/sqrt(f)'] = g.y / np.sqrt(g.f)
    print('==', nm); print(g.round(4).to_string())
    ff = df.f[df.f > 0].to_numpy(); yy = df.y[df.f > 0].to_numpy()
    for e in [0.4, 0.5, 0.6, 0.7]:
        x = ff ** e; c = (x * yy).sum() / (x * x).sum(); res = yy - c * x
        print('  e %.1f c %.4f resid var %.5f' % (e, c, res.var()))
    # with intercept
    X = np.column_stack([np.ones_like(ff), np.sqrt(ff)]); b, *_ = np.linalg.lstsq(X, yy, rcond=None)
    print('  a + c*sqrt(f): a %.4f c %.4f resid var %.5f' % (b[0], b[1], (yy - X @ b).var()))
    # unfilled IOCs vs ordered size
    u = df[df.f == 0]
    print('  unfilled takers n %d mean sign*dS %.4f' % (len(u), u.y.mean()))
