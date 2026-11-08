"""Inserts to exclude from the S likelihood: ladder chains (exact 2.000 ms same-side maker links) and the reactive
15-lot maker (maker insert 1..W us after an IOC taker). -> _cache/ladder_member.npy (lid, rung), _cache/react.npy (bool)."""
import numpy as np
import prep
d = prep.build()
us = d['t_us'].astype(np.int64); ioc = d['ioc']; s = d['sign']; n = len(us); v = d['volume']; p = d['price']
nxt = np.full(n, -1); prv = np.full(n, -1)
for off in (1999, 2000, 2001):
    j = np.searchsorted(us, us + off); ok = j < n; j = np.minimum(j, n - 1)
    ok &= (us[j] == us + off) & (s[j] == s) & ~ioc & ~ioc[j] & (nxt < 0)
    nxt[ok] = j[ok]
prv[nxt[nxt >= 0]] = np.flatnonzero(nxt >= 0)
heads = np.flatnonzero((nxt >= 0) & (prv < 0))
lid = np.full(n, -1); rung = np.full(n, -1)
for k, h in enumerate(heads):
    m = [h]
    while nxt[m[-1]] >= 0: m.append(nxt[m[-1]])
    lid[m] = k; rung[m] = np.arange(len(m))
np.save(prep.common.CACHE / 'ladder_member.npy', np.c_[lid, rung])
print('ladders', len(heads), 'members', (lid >= 0).sum())
# reactive: maker insert within 1..100 us after the latest IOC taker
tk = np.flatnonzero(ioc)
last = np.maximum.accumulate(np.where(ioc, np.arange(n), -1))
prevtk = np.r_[-1, last[:-1]]
dtk = np.where(prevtk >= 0, us - us[np.maximum(prevtk, 0)], 10**9)
for W in (1, 2, 5, 100):
    m = ~ioc & (dtk >= 1) & (dtk <= W)
    print('W', W, 'n', m.sum(), 'vol15', (m & (v == 15)).sum(), 'opp side', (m & (s != s[np.maximum(prevtk, 0)])).sum())
react = ~ioc & (dtk == 1)
react |= ~ioc & (dtk <= 100) & (v == 15)
np.save(prep.common.CACHE / 'react.npy', react)
print('react excluded', react.sum())
print('15-lot makers overall', ((v == 15) & ~ioc).sum(), 'neighbours 14,16', ((v == 14) & ~ioc).sum(), ((v == 16) & ~ioc).sum())
