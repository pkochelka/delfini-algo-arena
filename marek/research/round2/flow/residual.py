"""After masking shock windows: dispersion, ACF, and taker rate vs recent volatility (|dmid|)."""
import numpy as np, pandas as pd
from pathlib import Path
D = Path(__file__).parent; C = D / 'cache'
r = pd.read_pickle(C / 'rows.pkl'); ev = pd.read_csv(D / 'shock_events.csv')
ins = r[r.op == 0]
T = 6000; w = 1.0; n = int(T / w)
def cnt(t): return np.bincount((np.asarray(t) / w).astype(int), minlength=n)[:n].astype(float)
mk = cnt(ins.t[~ins.ioc]); tk = cnt(ins.t[ins.ioc]); tb = cnt(ins.t[ins.ioc & (ins.side == 0)]); ts = tk - tb
mid = ((r.bid_after + r.ask_after) / 2).ffill().to_numpy(); tt = r.t.to_numpy()
grid = np.arange(n + 1) * w
m_at = mid[np.clip(np.searchsorted(tt, grid, 'right') - 1, 0, None)]
dm = np.diff(m_at)  # mid change over bin i
# path variation within bin: sum |dmid| over events
dmid_ev = np.abs(np.diff(mid, prepend=mid[0])); dmid_ev[np.isnan(dmid_ev)] = 0
pv = np.bincount((tt / w).astype(int), weights=dmid_ev, minlength=n)[:n]
mask = np.ones(n, bool)
for o in ev.onset: mask[int(o - 2):int(o + 20)] = False
print('kept bins', mask.sum())
for name, x in [('maker', mk), ('taker', tk), ('tbuy', tb), ('tsell', ts)]:
    y = x[mask]; print(f'{name:6s} mean {y.mean():.2f} DI1s {y.var()/y.mean():.3f}', end=' ')
    for agg in [10, 60]:
        z = np.add.reduceat(np.where(mask, x, np.nan), np.arange(0, n, agg)); z = z[~np.isnan(z)]
        print(f'DI{agg}s {z.var()/z.mean():.2f}', end=' ')
    print()
def acf(x, L):
    y = np.where(mask, x, np.nan); y = y - np.nanmean(y)
    return [np.nanmean(y[:-l] * y[l:]) / np.nanvar(y) for l in L]
L = [1, 2, 3, 5, 10, 30, 60, 120]
print('lags', L)
for name, x in [('maker', mk), ('taker', tk), ('|dm|', np.abs(dm)), ('pathvar', pv)]: print(f'acf {name:7s}', np.round(acf(x, L), 3))
# cross-corr taker count vs lagged path variation and |dm|
def xc(a, b, l):
    ok = mask.copy()
    if l > 0: A, B, ok = a[:-l], b[l:], mask[:-l] & mask[l:]
    elif l < 0: A, B, ok = a[-l:], b[:l], mask[-l:] & mask[:l]
    else: A, B = a, b
    return np.corrcoef(A[ok], B[ok])[0, 1]
print('corr(pathvar[t], taker[t+l]) for l=-5..10:', [round(xc(pv, tk, l), 3) for l in range(-5, 11)])
print('corr(pathvar[t], maker[t+l]) for l=-5..10:', [round(xc(pv, mk, l), 3) for l in range(-5, 11)])
print('corr(taker[t], maker[t+l]) for l=-5..10:', [round(xc(tk, mk, l), 3) for l in range(-5, 11)])
print('corr(maker[t], maker[t+l])', [round(xc(mk, mk, l), 3) for l in range(0, 8)])
np.savez(C / 'bins1s.npz', mk=mk, tk=tk, tb=tb, ts=ts, dm=dm, pv=pv, mask=mask)
