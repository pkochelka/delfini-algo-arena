"""Per-0.1s / 1s bin counts per stream + maker size stats -> cache/bins.npz."""
import numpy as np, pandas as pd
from pathlib import Path
C = Path(__file__).parent / 'cache'
r = pd.read_pickle(C / 'rows.pkl'); ins = r[r.op == 0]
mk = ins[~ins.ioc]; tk = ins[ins.ioc]
T = 6000
out = {}
for w, name in [(0.1, '01'), (1.0, '1')]:
    nb = int(round(T / w))
    def cnt(t, wt=None): return np.bincount((np.asarray(t) / w).astype(int), weights=wt, minlength=nb)[:nb]
    out['mb' + name] = cnt(mk.t[mk.side == 0]); out['ma' + name] = cnt(mk.t[mk.side == 1])
    out['tb' + name] = cnt(tk.t[tk.side == 0]); out['ts' + name] = cnt(tk.t[tk.side == 1])
    out['mlogv' + name] = cnt(mk.t, np.log(mk.vol.to_numpy()))
    out['cx' + name] = cnt(r.t[r.op == 1])
np.savez(C / 'bins.npz', **out)
m = out['mb1'] + out['ma1']; tkc = out['tb1'] + out['ts1']
print('maker /s mean %.2f sd %.2f; taker /s mean %.3f' % (m.mean(), m.std(), tkc.mean()))
m60 = m.reshape(100, 60).sum(1); t60 = tkc.reshape(100, 60).sum(1)
lv = out['mlogv1'].reshape(100, 60).sum(1) / m60
mt = mk.t.to_numpy(); mv = mk.vol.to_numpy()
med = [np.median(mv[(mt >= i * 60) & (mt < i * 60 + 60)]) for i in range(100)]
for i in range(100): print(i, int(m60[i]), int(t60[i]), round(lv[i], 3), med[i], '#' * int((m60[i] - 10000) / 200))
