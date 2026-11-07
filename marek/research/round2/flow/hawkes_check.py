"""Short-range self-excitation tests: conditional intensity after an event vs local rate (+-5 s, excluding +-0.5 s)."""
import numpy as np, pandas as pd
from pathlib import Path
D = Path(__file__).parent; C = D / 'cache'
r = pd.read_pickle(C / 'rows.pkl'); ev = pd.read_csv(D / 'shock_events.csv')
ins = r[r.op == 0]
ok = np.ones(len(ins), bool); t_all = ins.t.to_numpy()
for o in ev.onset: ok &= ~((t_all > o - 2) & (t_all < o + 20))
I = ins[ok]
streams = {'TB': I.t[I.ioc & (I.side == 0)].to_numpy(), 'TS': I.t[I.ioc & (I.side == 1)].to_numpy(),
           'MB': I.t[~I.ioc & (I.side == 0)].to_numpy(), 'MA': I.t[~I.ioc & (I.side == 1)].to_numpy()}
win = [(0, 0.01), (0.01, 0.1), (0.1, 0.5), (0.5, 2)]
def ratio(src, dst):
    s = src[(src > 10) & (src < 5990)]
    out = []
    for a, b in win:
        n_after = np.searchsorted(dst, s + b) - np.searchsorted(dst, s + a, 'right')
        # local baseline: dst count in [s-5, s-2] U [s+2+? ...] use (s-6, s-3)+(s+3,s+6) excluding post
        base = (np.searchsorted(dst, s - 3) - np.searchsorted(dst, s - 6) + np.searchsorted(dst, s + 6) - np.searchsorted(dst, s + 3)) / 6
        out.append(n_after.sum() / (base.sum() * (b - a)))
    return out
print('ratio of dst intensity in window after src event to local baseline (±3–6 s); windows', win)
for a in streams:
    for b in streams:
        if a[0] == 'M' and b[0] == 'M' and a != b: pass
        print(f'{a}->{b}', np.round(ratio(streams[a], streams[b]), 3))
