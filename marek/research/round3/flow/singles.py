"""Hidden length-1 ladder orders: non-ladder maker counts per size band vs the round-2 base law
round(lognormal(2.657, 1.803)) at 180 /s, per regime (deep interiors, shocks masked)."""
import numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
D = Path(__file__).parent; C = D / 'cache'
r = pd.read_pickle(C / 'rows.pkl'); ins = r[r.op == 0].reset_index(drop=True); mem = np.load(C / 'ladder_member.npy')
R = pd.read_csv(D / 'regimes.csv'); ev = pd.read_csv(D / 'shock_events.csv'); L = pd.read_pickle(C / 'ladders.pkl')
ti = ins.t.to_numpy(); v = ins.vol.to_numpy(); base = ~ins.ioc.to_numpy() & (mem[:, 0] < 0)
lab = np.zeros(6000 * 10, int) - 1
for _, s in R.iterrows(): lab[int(np.ceil((s.start + 3) * 10)):int((s.end - 3) * 10)] = 1 if s.regime == 'volatile' else 0
for o in ev.onset: lab[int((o - 1) * 10):int((o + 10) * 10)] = -1
bands = [1, 10, 19, 30, 45, 70, 89, 130, 200, 320, 600, 10**7]
def P(lo, hi, mu=2.657, sd=1.803):
    a = -np.inf if lo == 1 else (np.log(lo - 0.5) - mu) / sd; b = (np.log(hi + 0.5) - mu) / sd; return stats.norm.cdf(b) - stats.norm.cdf(a)
for reg in [0, 1]:
    T = (lab == reg).sum() / 10; m = base & (lab[(ti * 10).astype(int)] == reg)
    lm = (lab[(L.t0.to_numpy() * 10).astype(int)] == reg)
    print(f'=== regime {"vol" if reg else "calm"}: T {T:.0f}s, base makers {m.sum() / T:.2f}/s, multi-rung ladders {lm.sum() / T:.2f}/s')
    tot = 0
    for lo, hi in zip(bands[:-1], bands[1:]):
        lo_ = lo if lo == 1 else lo + 1
        obs = ((v >= lo_) & (v <= hi) & m).sum() / T; exp = 180 * P(lo_, hi); tot += obs - exp
        print(f'  {lo_:5d}-{hi:8d}: obs {obs:7.2f} exp(180·r2 law) {exp:7.2f} excess {obs - exp:+6.2f}')
    print(f'  total excess {tot:+.2f}/s')
