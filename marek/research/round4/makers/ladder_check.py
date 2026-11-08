"""Round-4 ladder vs round 3: burst size law by regime (S v0 vol flag), rung noise, band (start level vs same-side
touch), length distribution, max live ladder orders per level (agent-count cap)."""
import numpy as np, pandas as pd
from scipy import stats
from s_load import with_s
o = with_s()
lad = o[(o.cls == 1)].copy()
lad['ls'] = np.log(lad.volume)
lad['v'] = lad.st == 'vol'
rel = np.where(lad.sg > 0, lad.price - lad.bid, lad.ask - lad.price)
lad['rel'] = rel
for v in (False, True):
    s = lad[lad.v == v]
    b = s.groupby('chain').agg(m=('ls', 'mean'), L=('L', 'first'))
    print('VOL' if v else 'CALM', 'bursts', len(b), 'rungs', len(s))
    print('  burst mean log size: mean %.3f sd %.3f skew %.2f' % (b.m.mean(), b.m.std(), stats.skew(b.m)), ' median B %.0f' % np.exp(b.m.median()))
    r = s.volume / s.groupby('chain').volume.transform('mean')
    print('  size/burst-mean quantiles 0.5,1,5,50,95,99,99.5 %', np.percentile(r, [0.5, 1, 5, 50, 95, 99, 99.5]).round(3))
    mx = s.groupby('chain').volume.agg(lambda x: x.max() / x.min())
    for L in (4, 6):
        q = mx[b.L == L]
        print(f'  L={L} max/min quantiles 50/90/99', np.percentile(q, [50, 90, 99]).round(2), 'n', len(q))
    st0 = s[s.k == 0]
    print('  start level share', (st0.rel.value_counts(normalize=True).sort_index(ascending=False).round(3)).to_dict())
    print('  length share', (b.L.value_counts(normalize=True).sort_index().round(3)).to_dict())
    print('  rung level range', s.rel.min(), s.rel.max(), 'share within -7..-2', s.rel.between(-7, -2).mean().round(3))
# U(a, b) rung noise: fit by max/min ratio sim for L=6
rng = np.random.default_rng(0)
for lo, hi in [(0.75, 1.25), (0.7, 1.3), (0.8, 1.2)]:
    u = rng.uniform(lo, hi, (20000, 6)) * 150
    u = np.round(u); print(f'sim U({lo},{hi}) L=6 max/min 50/90/99', np.percentile(u.max(1) / u.min(1), [50, 90, 99]).round(2))
# live ladder orders per (side, level) at ladder inserts: max count
lad = lad.sort_values('t_us')
t = lad.t_us.to_numpy(); e = lad.end_t_us.fillna(6.1e9).to_numpy(); p = lad.price.to_numpy(); sg = lad.sg.to_numpy()
cnt = []
idx = np.random.default_rng(1).choice(len(lad), 20000, replace=False)
for i in idx:
    j0 = np.searchsorted(t, t[i] - 60e6)
    m = (e[j0:i + 1] > t[i]) & (p[j0:i + 1] == p[i]) & (sg[j0:i + 1] == sg[i])
    cnt.append(m.sum())
print('live ladder orders at the same level incl. the new one:', pd.Series(cnt).value_counts().sort_index().to_dict())
