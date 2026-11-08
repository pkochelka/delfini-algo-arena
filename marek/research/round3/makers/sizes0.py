"""First look at maker sizes: overall lognormal fit, per-minute count/log-size, quantiles, round-number heaping."""
import numpy as np, pandas as pd
from prep import load
o = load(); m = o[~o.ioc.astype(bool)].copy()
ls = np.log(m.volume.to_numpy())
print('n makers', len(m), 'rate', len(m)/6000)
print('logsize mean %.4f sd %.4f skew %.4f kurt %.4f' % (ls.mean(), ls.std(), pd.Series(ls).skew(), pd.Series(ls).kurt()))
print('vol quantiles', np.percentile(m.volume, [1,5,10,25,50,75,90,95,99,99.9]), m.volume.max())
m['min'] = (m.t_us // 60_000_000).astype(int)
g = m.groupby('min').agg(n=('volume','size'), mu=('volume', lambda v: np.log(v).mean()), sd=('volume', lambda v: np.log(v).std()), med=('volume','median'), buy=('sg', lambda s: (s>0).mean()))
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 200)
print(g.to_string())
