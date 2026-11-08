"""Does P(buy)=Phi((S-mid)/sigma) explain the sign ACF? Compare observed ACF with the ACF of signs simulated from the fitted
law given the observed S-mid (conditionally independent), and the residual (s - m) ACF. Also unfilled-predecessor effect."""
import numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import norm
C = Path(__file__).parent / 'cache'
tk = pd.read_parquet(C / 'takers_s.parquet')
SIG = 4.42
x = (tk.s_hat - tk.mid).to_numpy(); m = 2 * norm.cdf(x / SIG) - 1; m[~np.isfinite(m)] = 0
s = tk.sg.to_numpy().astype(float); reg = tk.reg.to_numpy(); f = tk.filled.to_numpy()
ok = ((~tk.tw) & ((tk.shock_dt < -1) | (tk.shock_dt > 8))).to_numpy() & np.isfinite(x)
rng = np.random.default_rng(0)
def acf(a, L=10):
    a = a - a.mean(); return np.array([np.mean(a[:-k] * a[k:]) for k in range(1, L + 1)]) / a.var()
for R in [0, 1, 'all']:
    q = ok & ((reg == R) if R != 'all' else True)
    so, mo = s[q], m[q]
    sims = np.array([acf(np.where(rng.random(len(mo)) < (1 + mo) / 2, 1., -1.)) for _ in range(20)])
    print(f'regime {R}: obs ACF {np.round(acf(so), 3)}')
    print(f'          sim ACF {np.round(sims.mean(0), 3)} (sd {sims.std(0).mean():.3f})')
    print(f'          resid ACF {np.round(acf(so - mo), 3)}')
    i = np.arange(1, len(so)); fu = f[q]
    for nm, mm in [('prev unfilled', fu[i - 1] == 0), ('prev filled', fu[i - 1] > 0)]:
        print(f'          {nm}: obs corr {np.mean(so[i][mm] * so[i - 1][mm]):+.3f}  model E {np.mean(mo[i][mm] * so[i - 1][mm]):+.3f}')
