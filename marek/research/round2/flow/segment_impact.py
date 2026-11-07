"""Per regime segment: net mid move vs sum(sign*sqrt(filled)) and sum(sign*sqrt(ordered)); 10 s bins regression by regime."""
import sys
import numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from analysis.cache import load_tape
D = Path(__file__).parent; C = D / 'cache'
tape = load_tape('data/round2.csv', verbose=False); ev = tape.events; o = tape.orders
t = (ev.t_us.to_numpy() - ev.t_us.iloc[0]) / 1e6
bid = ev.bid.to_numpy().astype(float); ask = ev.ask.to_numpy().astype(float); bid[bid <= 0] = np.nan; ask[ask <= 0] = np.nan
mid = pd.Series((bid + ask) / 2).ffill().to_numpy()
def at(x): return mid[np.clip(np.searchsorted(t, x, 'right') - 1, 0, None)]
k = o[o.ioc.astype(bool)]; ti = t[k.row.to_numpy()]; sg = np.where(k.side.astype(str).to_numpy() == 'B', 1, -1)
uf = sg * np.sqrt(k.filled.to_numpy()); uo = sg * np.sqrt(k.volume.to_numpy())
shocks = pd.read_csv(D / 'shock_events.csv').onset.to_numpy()
hi = np.load(C / 'hmm_taker_gamma2.npy')[:, 1] > 0.5
w = 10; nb = 600
b = (ti // w).astype(int)
UF = np.bincount(b, uf, nb); UO = np.bincount(b, uo, nb)
dM = np.diff(at(np.arange(nb + 1) * w + 0.01)); dM = np.nan_to_num(dM)
reg = hi.reshape(nb, w).mean(1)
sh = np.zeros(nb, bool)
for s in shocks: sh[int((s - 1) // w):int((s + 15) // w) + 1] = True
for name, m in [('calm', (reg == 0) & ~sh), ('volatile', (reg == 1) & ~sh)]:
    for xn, X in [('filled', UF), ('ordered', UO)]:
        A = np.c_[X[m], np.ones(m.sum())]; coef, res, *_ = np.linalg.lstsq(A, dM[m], rcond=None)
        pred = A @ coef; r2 = 1 - ((dM[m] - pred) ** 2).sum() / ((dM[m] - dM[m].mean()) ** 2).sum()
        resid_sd = (dM[m] - pred).std()
        print(f'{name:8s} 10s bins n={m.sum():3d}  dmid ~ c*sum(sign*sqrt({xn})): c={coef[0]:.4f} R2={r2:.3f} resid sd {resid_sd:.2f} dmid sd {dM[m].std():.2f}')
