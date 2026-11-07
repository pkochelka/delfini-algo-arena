"""ACF / cross-correlation of counts in fine bins."""
import numpy as np, pandas as pd
from pathlib import Path
C = Path(__file__).parent / 'cache'
r = pd.read_pickle(C / 'rows.pkl')
ins = r[r.op == 0]; can = r[r.op == 1]
def cnt(t, w, T=6000): return np.bincount((np.asarray(t) / w).astype(int), minlength=int(T / w))[:int(T / w)].astype(float)
for w in [0.1, 1.0]:
    mb = cnt(ins.t[(~ins.ioc) & (ins.side == 0)], w); ma = cnt(ins.t[(~ins.ioc) & (ins.side == 1)], w)
    tb = cnt(ins.t[ins.ioc & (ins.side == 0)], w); ts = cnt(ins.t[ins.ioc & (ins.side == 1)], w)
    cc = cnt(can.t, w)
    def xc(a, b, L):
        a = a - a.mean(); b = b - b.mean(); n = len(a)
        return np.array([np.dot(a[:n - l], b[l:]) / np.sqrt(np.dot(a, a) * np.dot(b, b)) for l in L])
    L = [0, 1, 2, 3, 5, 10, 20, 50, 100, 200] if w == 1 else [0, 1, 2, 3, 5, 10, 20, 50]
    print(f'== bin {w}s, lags {L}')
    for name, a, b in [('mk_tot acf', mb + ma, mb + ma), ('tk_tot acf', tb + ts, tb + ts), ('mb x ma', mb, ma), ('tb x ts', tb, ts),
                       ('tk -> mk', tk := tb + ts, mb + ma), ('mk -> tk', mb + ma, tk), ('tb acf', tb, tb), ('cancel acf', cc, cc), ('mk->cancel', mb+ma, cc)]:
        print(f'  {name:12s}', np.round(xc(a, b, L), 3))
