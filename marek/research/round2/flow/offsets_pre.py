"""Taker offset vs envelope of maker inserts strictly BEFORE the taker (30 ms), and envelope width, by regime; plus R1 baseline."""
import sys
import numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
D = Path(__file__).parent; C = D / 'cache'
def run(t, op_ins, ioc, side, px, regime, label, win=0.03):
    mk = op_ins & ~ioc; tk = op_ins & ioc
    mt, mp, ms = t[mk], px[mk], side[mk]
    bt, bp = mt[ms == 0], mp[ms == 0]; at_, ap = mt[ms == 1], mp[ms == 1]
    # rolling max bid / min ask over (t-win, t): brute force via searchsorted + reduceat
    tt, tp, ts = t[tk], px[tk], side[tk]
    def env(T, P, fn):
        a = np.searchsorted(T, tt - win); b = np.searchsorted(T, tt)
        out = np.full(len(tt), np.nan)
        for i in np.where(b > a)[0]: out[i] = fn(P[a[i]:b[i]])
        return out
    mb = env(bt, bp, np.max); ma = env(at_, ap, np.min)
    M = (mb + ma) / 2; W = ma - mb; sg = np.where(ts == 0, 1, -1); off = sg * (tp - M)
    reg = regime(tt)
    for g, m in reg.items():
        ok = m & np.isfinite(off)
        print(f'{label} {g:9s} n={ok.sum():6d} width mean {np.nanmean(W[ok]):.2f} | taker offset mean {off[ok].mean():.3f} median {np.median(off[ok]):.2f} '
              f'| dist', np.round(np.histogram(np.clip(off[ok], -3, 6), np.arange(-3.25, 6.5, 0.5))[0] / ok.sum(), 3)[4:16])
from analysis.cache import load_tape
for name in ['round1', 'round2']:
    ev = load_tape(f'data/{name}.csv', verbose=False).events
    t = (ev.t_us.to_numpy() - ev.t_us.iloc[0]) / 1e6
    ins = (ev.op == 'I').to_numpy(); ioc = (ev.lifespan == 'F').to_numpy(); side = (ev.side == 'A').to_numpy().astype(int); px = ev.price.to_numpy()
    if name == 'round1':
        run(t, ins, ioc, side, px, lambda x: {'all': np.ones(len(x), bool)}, 'R1')
    else:
        hi = np.load(C / 'hmm_taker_gamma2.npy')[:, 1] > 0.5; sh_on = pd.read_csv(D / 'shock_events.csv').onset.to_numpy()
        def reg(x):
            sh = np.zeros(len(x), bool)
            for o in sh_on: sh |= (x > o - 0.5) & (x < o + 15)
            h = hi[x.astype(int).clip(0, 5999)]
            return {'calm': ~sh & ~h, 'volatile': ~sh & h}
        run(t, ins, ioc, side, px, reg, 'R2')
print('offset bins (centres):', np.arange(-3, 6.01, 0.5)[4:16])
