"""Is persistence an intensity effect (Hawkes / meta-order: extra same-side takers) or a side-probability effect (latent
direction, total rate flat)? Conditional intensity after a taker, same vs opposite side, by regime; count dispersion."""
import numpy as np, pandas as pd
from pathlib import Path
C = Path(__file__).parent / 'cache'
tk = pd.read_parquet(C / 'takers.parquet')
vol = np.load(C / 'vol1s.npy')
t = tk.t.to_numpy(); s = tk.sg.to_numpy(); ex = (np.abs(tk.shock_dt) > 30).to_numpy()
edges = np.array([0, .002, .01, .05, .1, .25, .5, 1, 2, 4, 8])
for reg in [0, 1]:
    secs = (vol == reg)
    # baseline per-side rate in this regime, ex shock
    sm = ex & (tk.vol1s.to_numpy() == reg)
    T = secs.sum() - 0  # approx
    base = sm.sum() / 2 / secs.sum()
    same = np.zeros(len(edges) - 1); opp = np.zeros(len(edges) - 1); N = 0
    idx = np.flatnonzero(sm)
    for i in idx:
        j = np.searchsorted(t, t[i] + edges[-1])
        if int(t[i] + edges[-1]) >= 6000 or vol[min(int(t[i] + edges[-1]), 5999)] != reg: continue
        d = t[i + 1:j] - t[i]; b = np.searchsorted(edges, d, 'right') - 1; sa = s[i + 1:j] == s[i]
        same += np.bincount(b[sa], minlength=len(edges) - 1)[:len(edges) - 1]; opp += np.bincount(b[~sa], minlength=len(edges) - 1)[:len(edges) - 1]; N += 1
    w = np.diff(edges) * N
    print(f'regime {reg}: base per side {base:.2f}/s, N={N}')
    for a, b_, x, y in zip(edges[:-1], edges[1:], same / w / base, opp / w / base):
        print(f'  {a:5.3f}-{b_:5.3f}  same x{x:5.2f}  opp x{y:5.2f}  total x{(x + y) / 2:5.2f}')
# dispersion of per-second counts in interior of regimes
z = np.load(C / 'bins.npz')
for reg in [0, 1]:
    m = (vol == reg); m[1:] &= vol[:-1] == reg; m[:-1] &= vol[1:] == reg
    for nm in ['tb1', 'ts1']:
        x = z[nm][m]; print(reg, nm, 'mean', round(x.mean(), 2), 'DI', round(x.var() / x.mean(), 3))
    tot = (z['tb1'] + z['ts1'])[m]; imb = (z['tb1'] - z['ts1'])[m]
    print(reg, 'total DI', round(tot.var() / tot.mean(), 3), 'imbalance var/mean(total)', round(imb.var() / tot.mean(), 3))
    for W in [5, 10, 30]:
        tb = np.convolve(z['tb1'], np.ones(W), 'valid'); ts = np.convolve(z['ts1'], np.ones(W), 'valid')
        mm = np.convolve(vol == reg, np.ones(W), 'valid') == W
        print('   W', W, 'total DI', round((tb + ts)[mm].var() / (tb + ts)[mm].mean(), 3), 'imb var/total', round((tb - ts)[mm].var() / (tb + ts)[mm].mean(), 3))
