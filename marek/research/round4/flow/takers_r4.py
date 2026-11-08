"""Regular-taker rates per regime and side (non-TWAP, ex-shock -1..+8 s), homogeneity per 1000 s block, size law MLE
(rounded lognormal), and arrival rate vs |S-mid| on a 0.05 s grid."""
import numpy as np, pandas as pd
from pathlib import Path
from scipy.optimize import minimize
from scipy.stats import norm
D = Path(__file__).parent; C = D / 'cache'
tk = pd.read_parquet(C / 'takers_s.parquet'); R = pd.read_csv(D / 'regimes.csv'); ev = pd.read_csv(D / 'shock_events.csv')
g = np.arange(0, 6000, 0.05) + 0.025; reg = np.zeros(len(g), int)
for _, r in R[R.regime == 'volatile'].iterrows(): reg[(g >= r.start) & (g < r.end)] = 1
ok = np.ones(len(g), bool)
for _, e in ev.iterrows(): ok &= ~((g > e.t_jump - 4) & (g < e.t_jump + 8))
t = tk.t.to_numpy(); gi = np.minimum((t / 0.05).astype(int), len(g) - 1)
nt = ~tk.tw.to_numpy(); s = tk.sg.to_numpy()
for Rg in [0, 1]:
    T = (ok & (reg == Rg)).sum() * 0.05; m = nt & ok[gi] & (reg[gi] == Rg)
    print(f'regime {Rg}: T {T:.0f} s, buy {np.sum(m & (s > 0)) / T:.3f}/s sell {np.sum(m & (s < 0)) / T:.3f}/s, buy share {np.mean(s[m] > 0):.4f}')
    blk = [(np.sum(m & (t >= a) & (t < a + 1000)) / max((ok & (reg == Rg) & (g >= a) & (g < a + 1000)).sum() * 0.05, 1)) for a in range(0, 6000, 1000)]
    print('   per 1000 s block total rate', np.round(blk, 2))
    mt = tk.tw.to_numpy() & ok[gi] & (reg[gi] == Rg); print(f'   TWAP children {mt.sum() / T:.3f}/s')
v = tk.vol.to_numpy()[nt]
def nll(b):
    mu, sg = b; lo = np.where(v <= 1, -np.inf, np.log(v - 0.5)); hi = np.log(v + 0.5)
    return -np.sum(np.log(np.maximum(norm.cdf((hi - mu) / sg) - norm.cdf((lo - mu) / sg), 1e-300)))
b = minimize(nll, [4, 1.4], method='Nelder-Mead').x; print('size rounded-lognormal MLE (non-TWAP)', np.round(b, 4), 'n', len(v), 'n==160', (v == 160).sum())
for Rg in [0, 1]:
    vv = tk.vol.to_numpy()[nt & (tk.reg.to_numpy() == Rg)]; print(f'  reg {Rg} mean log size {np.log(vv).mean():.3f} sd {np.log(vv).std():.3f}')
# rate vs |S-mid|
z = np.load(C / 's_series.npz'); r = pd.read_pickle(C / 'rows.pkl'); rt = r.t.to_numpy(); mid = ((r.bid_after + r.ask_after) / 2).ffill().to_numpy()
xg = z['s'][np.maximum(np.searchsorted(z['t'], g) - 1, 0)] - mid[np.maximum(np.searchsorted(rt, g) - 1, 0)]
cnt = np.bincount(gi[nt], minlength=len(g))
for Rg in [0, 1]:
    q = ok & (reg == Rg) & np.isfinite(xg)
    print(f'reg {Rg} taker rate (/s) by |S-mid| at grid:', {f'{a}-{b}': round(cnt[q & (np.abs(xg) >= a) & (np.abs(xg) < b)].mean() / 0.05, 2) for a, b in [(0, .25), (.25, .5), (.5, 1), (1, 2), (2, 4), (4, 99)]})
