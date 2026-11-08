"""Poisson GLM of ladder starts (both sides pooled, 10 ms grid, shocks masked): log-rate on regime + EWMA of |dS|
(kernels tau) or EWMA of taker count / touch changes. Which driver explains the rate, and does regime remain?"""
import numpy as np, pandas as pd
from pathlib import Path
import statsmodels.api as sm
D = Path(__file__).parent; C = D / 'cache'
r = pd.read_pickle(C / 'rows.pkl'); ins = r[r.op == 0].reset_index(drop=True)
S = pd.read_parquet(D.parent / 's_dynamics/s_path.parquet'); L = pd.read_pickle(C / 'ladders.pkl')
ti = ins.t.to_numpy(); s = S.s_hat.to_numpy(); vol = S.vol.to_numpy(); shock = S.shock.to_numpy()
W = 0.01; NB = int(6000 / W)
dS = np.abs(np.diff(s, prepend=s[0])); b = (ti / W).astype(int)
A = np.bincount(b, dS, NB)  # |dS| per bin
TK = np.bincount(b[ins.ioc.to_numpy()], minlength=NB).astype(float)
rt = r.t.to_numpy(); bidc = np.r_[0, np.diff(r.bid_after.to_numpy()) != 0] | np.r_[0, np.diff(r.ask_after.to_numpy()) != 0]
TC = np.bincount((rt[bidc.astype(bool)] / W).astype(int), minlength=NB)[:NB].astype(float)
y = np.bincount((L.t0.to_numpy() / W).astype(int), minlength=NB)[:NB]
kk = np.minimum(np.searchsorted(ti, np.arange(NB) * W) , len(ti) - 1)
V = vol[kk]; H = shock[kk].copy()
for o in pd.read_csv(D / 'shock_events.csv').onset: H[int((o - 1) / W):int((o + 8) / W)] = True
def ewma(x, tau):
    a = np.exp(-W / tau); out = np.empty_like(x); acc = 0.0
    from numba import njit
    return _ew(x, a)
from numba import njit
@njit(cache=True)
def _ew(x, a):
    out = np.empty_like(x); acc = 0.0
    for i in range(len(x)):
        out[i] = acc  # strictly past
        acc = acc * a + x[i]
    return out
m = ~H
def fit(cols, names):
    X = sm.add_constant(np.column_stack(cols)[m]); res = sm.GLM(y[m], X, family=sm.families.Poisson()).fit()
    return res
base = fit([V.astype(float)], ['vol']); print('regime only: llf %.1f coef %s' % (base.llf, np.round(base.params, 3)))
for tau in [0.05, 0.1, 0.25, 0.5, 1, 2, 5]:
    eA = np.log1p(_ew(A, np.exp(-W / tau)) * W / tau)  # ~ |dS| per s, log1p
    eT = np.log1p(_ew(TC, np.exp(-W / tau)) * W / tau)
    r1 = fit([V.astype(float), eA], 0); r2 = fit([eA], 0); r3 = fit([V.astype(float), eT], 0)
    print(f'tau {tau}: vol+log1p(|dS| rate) llf {r1.llf:.1f} coef {np.round(r1.params, 3)} | |dS| only llf {r2.llf:.1f} | vol+touch-change rate llf {r3.llf:.1f} coef {np.round(r3.params, 3)}')
