"""Informedness: taker sign vs past mid moves and vs future post-trade moves, round2 (calm/volatile) vs round1 baseline."""
import sys
import numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from analysis.cache import load_tape
D = Path(__file__).parent; C = D / 'cache'

def table(ev, ok_fn=None, label=''):
    t = (ev.t_us.to_numpy() - ev.t_us.iloc[0]) / 1e6
    bid = ev.bid.to_numpy().astype(float); ask = ev.ask.to_numpy().astype(float)
    bid[bid <= 0] = np.nan; ask[ask <= 0] = np.nan
    mid = pd.Series((bid + ask) / 2).ffill().to_numpy()
    isk = ((ev.op == 'I') & (ev.lifespan == 'F')).to_numpy()
    i = np.where(isk)[0]; sg = np.where(ev.side.to_numpy()[i] == 'B', 1, -1)
    ti = t[i]; pre = mid[i - 1]; post = mid[i]
    def at(x): return mid[np.clip(np.searchsorted(t, x, 'right') - 1, 0, None)]
    groups = ok_fn(ti) if ok_fn else {'all': np.ones(len(i), bool)}
    for gname, m in groups.items():
        out = [f'{label}{gname:9s} n={m.sum():6d}']
        for h in [1, 5, 30]:
            x = (pre - at(ti - h)); ok = m & np.isfinite(x); out.append(f'past{h}s corr {np.corrcoef(sg[ok], x[ok])[0, 1]:+.3f}')
        for h in [0.1, 1, 5]:
            f = sg * (at(ti + h) - post); out.append(f'fut{h}s {np.nanmean(f[m]):+.3f}')
        print(' | '.join(out))

t1 = load_tape('data/round1.csv', verbose=False)
table(t1.events, label='R1 ')
t2 = load_tape('data/round2.csv', verbose=False)
ev = pd.read_csv(D / 'shock_events.csv'); g2 = np.load(C / 'hmm_taker_gamma2.npy'); hi = g2[:, 1] > 0.5
def grp(ti):
    sh = np.zeros(len(ti), bool)
    for o in ev.onset: sh |= (ti > o - 0.5) & (ti < o + 15)
    h = hi[ti.astype(int).clip(0, 5999)]
    return {'calm': ~sh & ~h, 'volatile': ~sh & h, 'shock': sh}
table(t2.events, grp, label='R2 ')
