"""Drought onsets (maker insert rate collapse) -> _cache/droughts.csv. Provisional until flow agent publishes."""
import numpy as np, pandas as pd
import prep
d = prep.build(); t = d['t']; mk = ~d['ioc']
tm = t[mk]
edges = np.arange(0, 6000.0001, 0.05)
c, _ = np.histogram(tm, edges)
# rolling 0.5 s count
r = np.convolve(c, np.ones(10), 'valid')  # r[j] = count in [edges[j], edges[j]+0.5)
low = r < 50
rows = []
j = 0
while j < len(low):
    if low[j]:
        k = j
        while k < len(low) and low[k]: k += 1
        rows.append((edges[j], edges[k] + 0.45, (k - j) * 0.05))
        j = k
    else:
        j += 1
X = pd.DataFrame(rows, columns=['a', 'b', 'len'])
X = X[X.len > 1.0].reset_index(drop=True)
# refine onset: last maker insert before the gap structure: time where instantaneous rate drops -> use first 0.05 bin with count<=1 after a-0.5
on = []
for a in X.a:
    i0 = np.searchsorted(tm, a - 0.3); i1 = np.searchsorted(tm, a + 1.5)
    g = np.diff(tm[i0:i1]); k = np.argmax(g > 0.03)  # first gap > 30 ms
    on.append(tm[i0 + k])
X['onset'] = on
# drought end: rate back to normal
X['dur_est'] = X.b - X.onset
print(X.round(3).to_string()); print(len(X))
X.to_csv(prep.common.CACHE / 'droughts.csv', index=False)
