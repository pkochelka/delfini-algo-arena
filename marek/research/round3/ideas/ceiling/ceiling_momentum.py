"""Ceiling angle: does the book's lag behind S predict S? E[S(t+h) - S(t) | S - mid] per regime (0.1 s grid, shock windows out)."""
import sys, pathlib
import numpy as np, pandas as pd
H = pathlib.Path(__file__).parent
sys.path.insert(0, str(H.parents[2] / 'strategy'))
import eval3
ROOT = H.parents[5]
on, reg = eval3.load_ctx('data/round3.csv')
z = np.load(ROOT / 'data/.cache/round3_S_oracle.npz'); TI, SA = z['t_us'], z['S']
tp = pd.read_csv(ROOT / 'data/round3.csv', usecols=['Time'])
top = np.load(H.parents[2] / 'novelty/cache/round3_top.npz')
tt = np.round(tp.Time.values * 1e6).astype(np.int64)
g = np.arange(1e6, 5990e6, 0.1e6).astype(np.int64)
def at(T, A, t):
    return A[np.maximum(np.searchsorted(T, t, side='right') - 1, 0)]
bid = at(tt, top['bid'], g).astype(float); ask = at(tt, top['ask'], g).astype(float)
ok = (bid > 0) & (ask > 0)
S0 = at(TI, SA, g); mid = (bid + ask) / 2
lab = np.array([eval3.label(x * 1e-6, on, reg) for x in g])
x = S0 - mid
for r in ['calm', 'volatile']:
    m = ok & (lab == r)
    print(r, 'S-mid sd %.2f' % x[m].std())
    for h in [0.2, 1, 3, 10, 30]:
        y = at(TI, SA, g + int(h * 1e6)) - S0
        mm = m & (np.abs(y) < 20)
        b = np.polyfit(x[mm], y[mm], 1)[0]
        # bin means
        bins = [-9, -3, -2, -1, -0.5, 0.5, 1, 2, 3, 9]
        c = pd.cut(x[mm], bins)
        bm = pd.Series(y[mm]).groupby(c, observed=False).mean().round(2).tolist()
        print(f'  h {h:4}s beta {b:.3f}  E[dS | S-mid bin {bins}] = {bm}')
    # buy at ask when S - ask > -k, hold h: mean S(t+h) - ask
    for h in [1, 10]:
        y = at(TI, SA, g + int(h * 1e6))
        for lo in [-0.5, 0.0, 0.3]:
            mm = m & (S0 - ask > lo) & (S0 - ask <= lo + 0.5)
            print(f'  buy at ask with S-ask in ({lo},{lo+0.5}] (n={mm.sum()}): S(t+{h})-ask = {np.mean(y[mm]-ask[mm]):.2f}, S(t)-ask = {np.mean(S0[mm]-ask[mm]):.2f}')
