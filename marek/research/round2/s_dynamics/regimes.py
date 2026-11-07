import numpy as np, pandas as pd, sys
import prep
d = prep.build(); tag = sys.argv[1] if len(sys.argv) > 1 else 'v1'
W = float(sys.argv[2]) if len(sys.argv) > 2 else 10
r = dict(np.load(prep.common.CACHE / f'smooth_{tag}.npz'))
J = pd.read_csv(prep.common.CACHE / f'jumps_{tag}.csv')
t = d['t']; m = r['mean']; k = d['kind']; p = d['price']
nb = int(6000 / W)
b = np.minimum((t // W).astype(int), nb - 1)
df = pd.DataFrame(index=range(nb))
df['jumps'] = np.bincount(np.minimum((J.t // W).astype(int), nb - 1), minlength=nb)
df['jabs'] = np.bincount(np.minimum((J.t // W).astype(int), nb - 1), weights=np.abs(J['size']), minlength=nb)
df['takers'] = np.bincount(b[d['ioc']], minlength=nb)
df['makers'] = np.bincount(b[~d['ioc']], minlength=nb)
tk = (k >= 4) & (k <= 5)
x = d['sign'] * (p - m)
df['t_out'] = np.bincount(b[tk], weights=(x[tk] > 1.5), minlength=nb) / df.takers
mk = (k <= 1)
gap = np.where(d['sign'] > 0, m - p, p - m)
df['gap'] = np.bincount(b[mk], weights=gap[mk], minlength=nb) / np.bincount(b[mk], minlength=nb)
df['clip'] = np.bincount(b[~d['ioc']], weights=((k == 2) | (k == 3))[~d['ioc']], minlength=nb) / df.makers
df['bsz'] = np.bincount(b[~d['ioc']], weights=np.log(d['volume'][~d['ioc']]), minlength=nb) / df.makers
df['tsz'] = np.bincount(b[d['ioc']], weights=np.log(d['volume'][d['ioc']]), minlength=nb) / df.takers
g = np.arange(nb + 1) * W
i = np.clip(np.searchsorted(t, g) - 1, 0, len(t) - 1)
df['dS'] = np.diff(m[i])
print(df.corr().round(2).to_string())
pd.set_option('display.width', 250)
print(df.describe().round(2).to_string())
df.to_csv(prep.common.CACHE / f'regimes_{tag}_{int(W)}.csv')
