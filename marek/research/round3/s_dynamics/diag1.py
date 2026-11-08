import numpy as np, pandas as pd
import prep
d = prep.build(); r = dict(np.load(prep.common.CACHE / 'smooth_v1.npz'))
t = d['t']; k = d['kind']; p = d['price']; sg = d['sign']; m = r['mean']
sec = np.floor(t).astype(int)
mk = k <= 1  # unclipped makers
gap = sg * (m - p)  # S - bid, ask - S  (for makers)
tk = (k >= 4) & (k <= 5)
x = sg * (p - m)
df = pd.DataFrame(dict(sec=sec, mk=mk, close=mk & (gap < 0.7), tk=tk, x=np.where(tk, x, np.nan)))
g = df.groupby('sec').agg(nmk=('mk', 'sum'), nclose=('close', 'sum'), ntk=('tk', 'sum'), xmed=('x', 'median'))
g['fclose'] = g.nclose / g.nmk
print(g.fclose.describe()); print(np.histogram(g.fclose, bins=20, range=(0, 1)))
print(np.histogram(g.xmed.dropna(), bins=24, range=(0, 6)))
g.to_csv(prep.common.CACHE / 'diag1_sec.csv')
# print runs of fclose<0.3
lo = (g.fclose < 0.35).to_numpy()
s = g.index.to_numpy()
runs = []; i = 0
while i < len(lo):
    if lo[i]:
        j = i
        while j < len(lo) and lo[j]: j += 1
        runs.append((s[i], s[j - 1] + 1)); i = j
    else: i += 1
print(len(runs), runs)
