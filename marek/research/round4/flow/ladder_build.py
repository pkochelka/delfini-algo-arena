"""Build ladder chains from exact 2.000 ms links (same side, maker). -> cache/ladders.pkl (one row per ladder),
cache/ladder_member.npy (insert index -> ladder id or -1, rung index)."""
import numpy as np, pandas as pd
from pathlib import Path
D = Path(__file__).parent; C = D / 'cache'
r = pd.read_pickle(C / 'rows.pkl'); ins = r[r.op == 0].reset_index(drop=True)
reg = (np.load(C / 'hmm_joint_2.npy')[:, 1] > .5)
t = ins.t.to_numpy(); us = np.rint(t * 1e6).astype(np.int64); ioc = ins.ioc.to_numpy(); s = ins.side.to_numpy()
p = ins.px.to_numpy(); v = ins.vol.to_numpy(); n = len(t)
nxt = np.full(n, -1); prv = np.full(n, -1)
for off in (1999, 2000, 2001):
    j = np.searchsorted(us, us + off); ok = (j < n)
    j = np.minimum(j, n - 1); ok &= (us[j] == us + off) & (s[j] == s) & ~ioc & ~ioc[j] & (nxt < 0)
    nxt[ok] = j[ok]
prv[nxt[nxt >= 0]] = np.flatnonzero(nxt >= 0)
heads = np.flatnonzero((nxt >= 0) & (prv < 0))
lid = np.full(n, -1); rung = np.full(n, -1)
rows = []
for k, h in enumerate(heads):
    m = [h]
    while nxt[m[-1]] >= 0: m.append(nxt[m[-1]])
    m = np.array(m); lid[m] = k; rung[m] = np.arange(len(m))
    sgn = -1 if s[h] == 0 else 1
    rows.append(dict(lid=k, t0=t[h], side=s[h], n=len(m), p0=p[h], steps=tuple(((p[m[1:]] - p[m[:-1]]) * sgn).astype(int)),
                     sizes=tuple(v[m]), bid_before=ins.bid_before[h], ask_before=ins.ask_before[h], vol=reg[int(t[h])]))
L = pd.DataFrame(rows)
L.to_pickle(C / 'ladders.pkl'); np.save(C / 'ladder_member.npy', np.c_[lid, rung])
print('ladders', len(L), 'inserts in ladders', (lid >= 0).sum(), 'share of maker inserts', (lid >= 0).sum() / (~ioc).sum())
print('rungs per ladder:', L.n.value_counts().sort_index().to_dict())
print('by regime: count', L.groupby('vol').size().to_dict(), 'rate /s', (L.groupby('vol').size() / np.array([(~reg).sum(), reg.sum()])).round(3).to_dict())
print('rungs by regime'); print(L.groupby('vol').n.value_counts().unstack(0))
print('side', L.side.mean())
allsteps = np.concatenate([np.array(x) for x in L.steps]); print('steps', dict(zip(*np.unique(allsteps, return_counts=True))))
