"""Impact per sqrt(filled) by regime: regress signed future mid change on sqrt(filled), for 2 and 5 s horizons; R1 baseline."""
import sys
import numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from analysis.cache import load_tape
D = Path(__file__).parent; C = D / 'cache'
def run(tape, groups_fn, label):
    ev = tape.events; o = tape.orders
    t = (ev.t_us.to_numpy() - ev.t_us.iloc[0]) / 1e6
    bid = ev.bid.to_numpy().astype(float); ask = ev.ask.to_numpy().astype(float); bid[bid <= 0] = np.nan; ask[ask <= 0] = np.nan
    mid = pd.Series((bid + ask) / 2).ffill().to_numpy()
    k = o[o.ioc.astype(bool)]
    i = k.row.to_numpy(); sg = np.where(k.side.astype(str).to_numpy() == 'B', 1, -1); q = k.filled.to_numpy().astype(float)
    ti = t[i]; pre = mid[np.maximum(i - 1, 0)]
    def at(x): return mid[np.clip(np.searchsorted(t, x, 'right') - 1, 0, None)]
    for g, m in groups_fn(ti).items():
        for h in [2, 5]:
            y = sg * (at(ti + h) - pre); x = np.sqrt(q); ok = m & np.isfinite(y)
            b = np.polyfit(x[ok], y[ok], 1)
            print(f'{label} {g:9s} h={h}s slope per sqrt(filled) {b[0]:.4f} intercept {b[1]:+.3f}  frac unfilled {np.mean(q[m]==0):.3f}  mean filled {q[m].mean():.1f} n {ok.sum()}')
run(load_tape('data/round1.csv', verbose=False), lambda ti: {'all': np.ones(len(ti), bool)}, 'R1')
ev = pd.read_csv(D / 'shock_events.csv'); hi = np.load(C / 'hmm_taker_gamma2.npy')[:, 1] > 0.5
def grp(ti):
    sh = np.zeros(len(ti), bool)
    for o in ev.onset: sh |= (ti > o - 0.5) & (ti < o + 15)
    h = hi[ti.astype(int).clip(0, 5999)]
    return {'calm': ~sh & ~h, 'volatile': ~sh & h}
run(load_tape('data/round2.csv', verbose=False), grp, 'R2')
