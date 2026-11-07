"""Is taker price still round(S±1)? Consistency with maker inserts in the few ms before the taker.
Buy at p => S in [p-1.5, p-0.5] => preceding maker bids <= p-1 and asks >= p-1. Sell at p => bids <= p+1, asks >= p+1.
Also: taker size distribution (rounded lognormal check P(v=k)), skew/kurtosis, by regime / shock burst."""
import numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import norm, skew, kurtosis
D = Path(__file__).parent; C = D / 'cache'
r = pd.read_pickle(C / 'rows.pkl'); ev = pd.read_csv(D / 'shock_events.csv')
hi = np.load(C / 'hmm_taker_gamma2.npy')[:, 1] > 0.5
ins = r[r.op == 0]
mk = ins[~ins.ioc]; tk = ins[ins.ioc]
mt = mk.t.to_numpy(); mpx = mk.px.to_numpy(); mside = mk.side.to_numpy()
for win in [0.002, 0.005, 0.02]:
    viol = 0; tot = 0; vt = []
    for t, p, s in zip(tk.t.to_numpy(), tk.px.to_numpy(), tk.side.to_numpy()):
        a = np.searchsorted(mt, t - win); b = np.searchsorted(mt, t)
        if b == a: continue
        P = mpx[a:b]; Sd = mside[a:b]
        ref = p - 1 if s == 0 else p + 1
        bad = np.any((Sd == 0) & (P > ref)) or np.any((Sd == 1) & (P < ref))
        tot += 1; viol += bad
        if bad: vt.append(t)
    vt = np.array(vt)
    sh = np.zeros(len(vt), bool)
    for o in ev.onset: sh |= (vt > o + 2.5) & (vt < o + 4)
    print(f'window {win*1e3:.0f} ms: takers with makers in window {tot}, violating {viol} ({viol/tot:.4%}); of violations near jumps {sh.sum()}, in volatile regime {hi[vt.astype(int)].sum() if len(vt) else 0}')
# size
v = tk.vol.to_numpy(); mu, sd = 4.1044, 1.4024
print('P(v=k) obs vs model:', [(k, round(np.mean(v == k), 5), round(norm.cdf((np.log(k + .5) - mu) / sd) - (norm.cdf((np.log(k - .5) - mu) / sd) if k > 1 else 0), 5)) for k in range(1, 7)])
lv = np.log(v[v >= 20]); print('log size (v>=20) skew %.3f' % skew(np.log(v)), 'kurt %.3f' % kurtosis(np.log(v)))
for thr in [1000, 5000, 10000, 20000]:
    print(f'  taker v>{thr}: obs {np.sum(v > thr)} exp {len(v) * (1 - norm.cdf((np.log(thr + .5) - mu) / sd)):.1f}')
rel = np.full(len(tk), False)
for o in ev.end: rel |= ((tk.t > o) & (tk.t < o + 3)).to_numpy()
print('log taker size: burst %.3f±%.3f (n %d), other %.3f' % (np.log(v[rel]).mean(), np.log(v[rel]).std() / np.sqrt(rel.sum()), rel.sum(), np.log(v[~rel]).mean()))
