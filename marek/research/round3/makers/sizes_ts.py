"""10 s time series of maker counts by size band: A band 110-250, B band 25-55, rest."""
import numpy as np, pandas as pd
from prep import load
o = load(); m = o[~o.ioc.astype(bool)].copy()
W = 10
m['w'] = (m.t_us // (W*1_000_000)).astype(int)
v = m.volume
m['A'] = v.between(110, 250); m['B'] = v.between(25, 55); m['small'] = v <= 10
g = m.groupby('w').agg(n=('volume','size'), A=('A','sum'), B=('B','sum'), small=('small','sum'))
g['n_s'] = g.n / W; g['fA'] = g.A/g.small; g['fB'] = g.B/g.small
g.to_csv('cache/sizes_ts10.csv')
pd.set_option('display.max_rows', 700)
print(g[['n_s','small','fA','fB']].round(3).to_string())
