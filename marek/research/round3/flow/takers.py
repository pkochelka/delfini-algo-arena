"""Taker basics: rates per side/regime/block, sizes (lognormal fit, tail), sign process, gaps, price vs touch,
future mid move after trade (impact vs information, proxy until S is available)."""
import numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
D = Path(__file__).parent; C = D / 'cache'
r = pd.read_pickle(C / 'rows.pkl'); ev = pd.read_csv(D / 'shock_events.csv')
reg = np.load(C / 'hmm_joint_2.npy')[:, 1] > 0.5
tt = r.t.to_numpy(); mid = ((r.bid_after + r.ask_after) / 2).ffill().to_numpy()
idx = np.where((r.op == 0) & r.ioc)[0]; k = r.iloc[idx].copy()
k['sg'] = np.where(k.side == 0, 1, -1); k['mid_pre'] = ((k.bid_before + k.ask_before) / 2).to_numpy(); k['mid_post'] = mid[idx]
k['touch'] = np.where(k.side == 0, k.ask_before, k.bid_before); k['off_touch'] = k.sg * (k.px - k.touch)
k['vol_reg'] = reg[k.t.astype(int).clip(0, 5999)]
sh = np.zeros(len(k), bool)
for o in ev.onset: sh |= ((k.t > o - 0.5) & (k.t < o + 15)).to_numpy()
k['shock'] = sh
pd.set_option('display.width', 200)
K = k[~k.shock]
secs = {False: ((~reg) & np.load(C / 'bins1s.npz')['mask']).sum(), True: (reg & np.load(C / 'bins1s.npz')['mask']).sum()}
for v in [False, True]:
    X = K[K.vol_reg == v]; print(f'regime vol={v}: buy /s {(X.side == 0).sum() / secs[v]:.3f} sell /s {(X.side == 1).sum() / secs[v]:.3f} buy share {(X.side == 0).mean():.4f}')
blk = (K.t // 1000).astype(int)
print('per 1000 s block, rate per side by regime:')
for b in range(6):
    s = np.load(C / 'bins1s.npz')['mask'].copy(); rr = reg.copy(); sel = np.zeros(6000, bool); sel[b * 1000:(b + 1) * 1000] = True
    out = []
    for v in [False, True]:
        n = ((blk == b) & (K.vol_reg == v)).sum(); T = (sel & s & (rr == v)).sum(); out.append(f'{n / T / 2 if T else np.nan:.2f} (T {T})')
    print('  block', b, out)
lv = np.log(k.vol.to_numpy())
print('log size: mean %.4f sd %.4f skew %.3f kurt %.3f' % (lv.mean(), lv.std(), stats.skew(lv), stats.kurtosis(lv)))
# rounded lognormal MLE
from scipy.optimize import minimize
v = k.vol.to_numpy()
def nll(th):
    mu, sd = th
    lo = np.where(v == 1, -np.inf, np.log(np.maximum(v - 0.5, 1e-9))); hi = np.log(v + 0.5)
    return -np.log(stats.norm.cdf((hi - mu) / sd) - stats.norm.cdf((lo - mu) / sd) + 1e-300).sum()
th = minimize(nll, [4.1, 1.4], method='Nelder-Mead').x; print('rounded lognormal MLE', th.round(4))
for q in [1000, 5000, 20000]:
    print(f'  P(v>{q}) obs {(v > q).sum()} exp {len(v) * stats.norm.sf((np.log(q + .5) - th[0]) / th[1]):.1f}')
print('log size by side/regime/shock:'); print(k.assign(lv=lv).groupby(['shock', 'vol_reg', 'side']).lv.agg(['mean', 'std', 'count']).round(3).to_string())
s = K.sg.to_numpy(); print('sign acf lags 1-5', [round(np.corrcoef(s[:-l], s[l:])[0, 1], 4) for l in range(1, 6)])
gap = np.diff(K.t.to_numpy()); same = s[1:] == s[:-1]
print('P(same side | gap):'); print(pd.Series(same).groupby(pd.cut(gap, [0, 0.001, 0.01, 0.05, 0.2, 1, 10])).agg(['mean', 'count']).round(3).to_string())
print('price - touch (signed):'); print(pd.crosstab(K.vol_reg, K.off_touch.clip(-3, 6), normalize='index').round(3).to_string())
def mid_at(x): return mid[np.clip(np.searchsorted(tt, x, 'right') - 1, 0, None)]
for h in [0.1, 1, 5, 30]:
    fut = mid_at(K.t.to_numpy() + h); d = K.sg * (fut - K.mid_post)
    print(f'h={h}: E[sg*(mid_h-mid_post)] calm {d[~K.vol_reg].mean():+.3f}±{d[~K.vol_reg].std() / np.sqrt((~K.vol_reg).sum()):.3f} vol {d[K.vol_reg].mean():+.3f}±{d[K.vol_reg].std() / np.sqrt(K.vol_reg.sum()):.3f}')
for h in [1, 5, 30]:
    past = K.mid_pre - mid_at(K.t.to_numpy() - h)
    print(f'corr(sign, past {h}s mid move): calm {np.corrcoef(K.sg[~K.vol_reg], past[~K.vol_reg])[0, 1]:+.4f} vol {np.corrcoef(K.sg[K.vol_reg], past[K.vol_reg])[0, 1]:+.4f}')
k.to_pickle(C / 'takers.pkl')
