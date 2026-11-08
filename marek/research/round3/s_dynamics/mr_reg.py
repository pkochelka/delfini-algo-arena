"""Mean reversion of S: per-second regression dS - impact = a_k - b_reg * S + noise.
Anchor hypotheses: interval fixed effects (F const between shocks), global const, none. Path TAG (np)."""
import sys, numpy as np, pandas as pd
import prep
tag = sys.argv[1] if len(sys.argv) > 1 else 'np'
e = 0.6
d = prep.build(); r = dict(np.load(prep.common.CACHE / f'smooth_{tag}.npz'))
t = d['t']; m = r['mean']; vol = r['vol'].astype(bool)
cc = np.where(vol, 0.065, 0.0149)
imp = np.where(d['ioc'], cc * d['sign'] * d['filled'] ** e, 0.0); C = np.cumsum(imp)
dt = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
g = np.arange(0, 6000, dt); i = np.clip(np.searchsorted(t, g, 'right') - 1, 0, len(t) - 1)
S = m[i]; Cg = C[i]; V = vol[i]
SH = pd.read_csv(prep.common.HERE / '../flow/shock_events.csv').onset.to_numpy()
seg = np.searchsorted(SH + 3.5, g)            # interval id: between jumps
bad = np.zeros(len(g), bool)
for a in SH: bad |= (g >= a - 1) & (g < a + 5)
y = (S[1:] - S[:-1]) - (Cg[1:] - Cg[:-1]); x = S[:-1]; sg = seg[:-1]; ok = ~bad[:-1] & (seg[1:] == seg[:-1])
vv = V[:-1] & V[1:]; cv = ~V[:-1] & ~V[1:]
df = pd.DataFrame(dict(y=y, x=x, s=sg, ok=ok, v=vv, c=cv))[ok]
def fe_fit(df, both=True):
    # within-interval demeaning (fixed effects) with separate slopes for calm / vol
    out = {}
    D = df.copy()
    D['xc'] = D.x * D.c; D['xv'] = D.x * D.v
    for col in ('y', 'xc', 'xv', 'c', 'v'):
        D[col + '_d'] = D[col] - D.groupby('s')[col].transform('mean')
    X = D[['xc_d', 'xv_d']].to_numpy(); Y = D.y_d.to_numpy()
    b, *_ = np.linalg.lstsq(X, Y, rcond=None)
    res = Y - X @ b
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * res.var())
    return b, se, res.var()
b, se, rv = fe_fit(df)
print('dt %.2f  FE(interval anchor): b_calm %.5f (se %.5f) -> T %.1f s ; b_vol %.5f (se %.5f) -> T %.1f s; resid var %.4f' % (dt, b[0], se[0], dt / -b[0] if b[0] < 0 else np.inf, b[1], se[1], dt / -b[1] if b[1] < 0 else np.inf, rv))
# global anchor
X = np.c_[np.ones(len(df)), df.x * df.c, df.x * df.v, df.v]; bb, *_ = np.linalg.lstsq(X, df.y, rcond=None)
print('  global anchor: b_calm %.5f b_vol %.5f, implied F_calm %.1f, resid var %.4f' % (bb[1], bb[2], -bb[0] / bb[1], (df.y - X @ bb).var()))
# per-interval implied anchors vs interval mean / start level
D = df.copy(); D['xd'] = D.x - D.groupby('s').x.transform('mean')
print('  pooled b (no FE)', np.polyfit(df.x, df.y, 1)[0])
