"""Write s_path.parquet from a smoothed run: one row per insert, S just before that insert's own impact.
Columns: t_us, s_lo, s_hi (99% smoothed credible interval), s_hat (posterior mean), s_sd, s_filt (causal filtered mean),
s_filt_sd, vol (regime label used), shock (shock-jump window)."""
import sys
import numpy as np, pandas as pd
import prep
tag = sys.argv[1] if len(sys.argv) > 1 else 'v3'
d = prep.build(); r = dict(np.load(prep.common.CACHE / f'smooth_{tag}.npz'))
df = pd.DataFrame(dict(t_us=d['t_us'], s_lo=r['lo'], s_hi=r['hi'], s_hat=r['mean'], s_sd=r['sd'], s_filt=r['fmean'],
                       s_filt_sd=r['fsd'], vol=r['vol'].astype(bool), shock=r['shock'].astype(bool)))
df.to_parquet(prep.common.HERE / 's_path.parquet', index=False)
print(df.describe().T.round(3))
