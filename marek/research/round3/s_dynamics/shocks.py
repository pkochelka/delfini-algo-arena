"""Shock events: exact jump time relative to drought onset, size, S drift in drought, post-jump behaviour."""
import numpy as np, pandas as pd, sys
import prep
tag = sys.argv[1] if len(sys.argv) > 1 else 'v1'
d = prep.build(); r = dict(np.load(prep.common.CACHE / f'smooth_{tag}.npz'))
t = d['t']; m = r['mean']; sd = r['sd']
SH = pd.read_csv(prep.common.ROOT / 'marek/research/round3/flow/shock_events.csv')
def at(x): return m[np.clip(np.searchsorted(t, x, 'right') - 1, 0, len(t) - 1)]
rows = []
mk = ~d['ioc']
for a in SH.onset:
    # refine onset: maker insert rate drop. find first time after a-1 when 0.25s window count < 20
    tm = t[mk]
    j0 = np.searchsorted(t, a + 2.0); j1 = np.searchsorted(t, a + 4.5)
    dm = np.diff(m[j0:j1]); k = np.argmax(np.abs(dm))
    tj = t[j0 + k + 1]  # S has new value at this event
    tprev = t[j0 + k]
    # makers in last 1 s before onset estimate & drought rate
    rows.append(dict(onset=a, t_jump=tj, t_prev=tprev, lag=tj - a, jump=at(tj + 0.05) - at(tprev - 0.05),
                     drift_drought=at(a + 2.9) - at(a), pre10=at(a) - at(a - 10), post10=at(tj + 10) - at(tj + 0.2),
                     post_1=at(tj + 1) - at(tj + 0.05), post30=at(tj + 30) - at(tj + 0.2), post60=at(tj + 60) - at(tj + 0.2), post120=at(tj + 120) - at(tj + 0.2), sd_jump=sd[j0 + k + 1]))
X = pd.DataFrame(rows)
pd.set_option('display.width', 200)
print(X.round(3).to_string())
print('jump sd %.1f mean %.1f mean|J| %.1f kurt %.2f' % (X.jump.std(), X.jump.mean(), X.jump.abs().mean(), X.jump.kurt()))
print('drought drift sd', X.drift_drought.std(), 'post10 sd', X.post10.std(), 'corr(jump,post10)', np.corrcoef(X.jump, X.post10)[0, 1], 'corr(jump,pre10)', np.corrcoef(X.jump, X.pre10)[0, 1])
X.to_csv(prep.common.CACHE / f'shocks_{tag}.csv', index=False)
for h in ('post10', 'post30', 'post60', 'post120'):
    b = np.polyfit(X.jump, X[h], 1)[0]
    print('%s: slope on jump %.3f  corr %.2f' % (h, b, np.corrcoef(X.jump, X[h])[0, 1]))
