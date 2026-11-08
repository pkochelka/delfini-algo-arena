"""Stop rule of the 15-lot agent: per instance displacement, fills, duration; S path shape within long instances."""
import numpy as np, pandas as pd
S = pd.read_parquet('../s_dynamics/s_path.parquet')
st = S.t_us.to_numpy(); sh = S.s_hat.to_numpy()
def s(t): return sh[np.clip(np.searchsorted(st, t, 'right') - 1, 0, len(st) - 1)]
a = pd.read_pickle('cache/r15_v2.pkl').sort_values('t_us')
rows = []
for ch, g in a.groupby('chain'):
    f, l = g.iloc[0], g.iloc[-1]
    sg = f.sg
    rows.append(dict(t0=f.t_us / 1e6, dur=(l.end_t_us - f.t_us) / 1e6, sg=sg, nfill=g.filled.sum() / 15, end=l.end,
                     p0=f.price, p1=l.price, dp=sg * (l.price - f.price), dS=sg * (s(l.end_t_us) - s(f.t_us)),
                     S0_rel=sg * (s(f.t_us) - f.price), S1_rel=sg * (s(l.end_t_us) - l.price),
                     maxp=(g.price * sg).max() * sg, n=len(g)))
r = pd.DataFrame(rows).sort_values('t0')
pd.set_option('display.width', 250)
print(r.round(2).to_string())
print('corr nfill~dur', np.corrcoef(r.nfill, r.dur)[0, 1].round(2), 'nfill~dp', np.corrcoef(r.nfill, r.dp)[0, 1].round(2),
      'dp~dur', np.corrcoef(r.dp, r.dur)[0, 1].round(2))
print('fills per second', (r.nfill / r.dur).describe().round(2).to_dict())
# within-instance S shape for long instances: S(t0 + f*dur) at fractions
L = r[r.dur > 6]
fr = np.linspace(0, 1, 11)
P = np.array([[x.sg * (s(x.t0 * 1e6 + q * x.dur * 1e6) - s(x.t0 * 1e6)) / max(abs(x.dS), 1e-9) * np.sign(x.dS) for q in fr] for x in L.itertuples()])
print('normalised S path at fractions of duration (long instances, n=%d):' % len(L), np.median(P, 0).round(2))
