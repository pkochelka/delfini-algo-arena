"""Per-taker impact (np smoother) vs candidate features: sqrt(filled), filled^0.6, fill-price location vs S, levels swept."""
import numpy as np, pandas as pd, sys
import prep
from common import load_tape
tag = sys.argv[1] if len(sys.argv) > 1 else 'np'
d = prep.build(); r = dict(np.load(prep.common.CACHE / f'smooth_{tag}.npz'))
T = load_tape(prep.common.ROOT / 'data/round2.csv')
tr = T.trades
rows = T.orders.row.to_numpy()
g = tr.groupby('taker_row').agg(last=('price', 'last'), first=('price', 'first'), q=('qty', 'sum'),
                               vw=('price', lambda s: 0), nlev=('price', 'nunique'), nfill=('qty', 'size'))
vw = (tr.price * tr.qty).groupby(tr.taker_row).sum() / tr.qty.groupby(tr.taker_row).sum(); g['vw'] = vw
m = r['mean']; vol = r['vol'].astype(bool); sh = r['shock'].astype(bool)
tk = np.where(d['ioc'] & (d['filled'] > 0))[0]; tk = tk[tk < len(m) - 1]
G = g.reindex(rows[tk])
s = d['sign'][tk]
df = pd.DataFrame(dict(y=s * (m[tk + 1] - m[tk]), f=d['filled'][tk], v=d['volume'][tk], last=s * (G['last'].to_numpy() - m[tk]),
                       first=s * (G['first'].to_numpy() - m[tk]), vw=s * (G.vw.to_numpy() - m[tk]), nlev=G.nlev.to_numpy(), nfill=G.nfill.to_numpy(),
                       reg=np.where(sh[tk], 'shock', np.where(vol[tk], 'vol', 'calm'))))
df['sq'] = np.sqrt(df.f); df['f6'] = df.f ** 0.6
for reg in ['calm', 'vol']:
    D = df[df.reg == reg]
    print('==', reg, len(D))
    for cols in [['sq'], ['f6'], ['sq', 'vw'], ['sq', 'last'], ['sq', 'first'], ['sq', 'nlev'], ['sq', 'nfill'], ['sq', 'last', 'nfill'], ['sq', 'f'], ['sq', 'vw', 'nfill', 'f']]:
        X = np.column_stack([np.ones(len(D))] + [D[c].to_numpy() for c in cols])
        b, *_ = np.linalg.lstsq(X, D.y.to_numpy(), rcond=None)
        res = D.y.to_numpy() - X @ b
        print('  ', cols, 'coef', np.round(b, 4), 'resid var %.5f' % res.var())
    print('   corr(y - c sqrt f, nfill)', np.corrcoef(D.y - 0.0281 * D.sq, D.nfill)[0, 1].round(3))
    print(D.groupby(D.nfill.clip(1, 8)).apply(lambda z: pd.Series(dict(n=len(z), f=z.f.mean(), ratio=(z.y / z.sq).mean()))).round(4).to_string())
