"""For the non-shock stale events (nonshock_NAME_thrX.pkl): our position before, and what r3_best took within 10 ms."""
import sys, numpy as np, pandas as pd
from common import *
name, thr, simlog = sys.argv[1], sys.argv[2], sys.argv[3]
R = pd.read_pickle(pathlib.Path(__file__).parent / f'nonshock_{name}_thr{thr}.pkl')
X = pd.read_pickle(pathlib.Path(__file__).parent / f'nonshock_{name}.pkl')
R['d'] = X.d.values[R.n.values]
St, Sv = s_path(name)
st = pd.read_csv(pathlib.Path(simlog) / 'trades.csv')
st = st[(st.taker_owner == 'user') | (st.maker_owner == 'user')].copy(); st['t'] = st.timestamp_us * 1e-6
st['sgn'] = np.where(((st.taker_owner == 'user') & (st.taker_side == 'B')) | ((st.maker_owner == 'user') & (st.taker_side == 'A')), 1, -1)
pos = np.cumsum(st.sgn.values * st.quantity.values); tt = st.t.values
S = Sv[np.clip(np.searchsorted(St, tt, 'right') - 1, 0, None)]
R['pos'] = [pos[i - 1] if i > 0 else 0 for i in np.searchsorted(tt, R.t.values - 0.005)]
# our same-direction fills in [t-5ms, t+10ms] (t = check time = taker + 5 ms)
a = np.searchsorted(tt, R.t.values - 0.005); b = np.searchsorted(tt, R.t.values + 0.010)
mq, mv = [], []
for i, (x, y, d) in enumerate(zip(a, b, R.d.values)):
    m = st.sgn.values[x:y] == d
    q = st.quantity.values[x:y][m]; e = (S[x:y][m] - st.price.values[x:y][m]) * d
    mq.append(q.sum()); mv.append((q * e).sum())
R['my_q'] = mq; R['my_v'] = mv
R['pinned'] = R.pos * R.d >= 95
for g, Y in R.groupby('vol'):
    print('vol' if g else 'calm', 'n', len(Y), 'V', round(Y.V.sum()), 'V100', round(Y.V100.sum()), 'V300', round(Y.V300.sum()), 'V950', round(Y.V950.sum()),
          'my_v', round(Y.my_v.sum()), 'my_q', Y.my_q.sum(), 'pinned share', round(Y.pinned.mean(), 2),
          'V100 when pinned', round(Y[Y.pinned].V100.sum()))
    print('   pos*d quantiles', np.percentile(Y.pos * Y.d, [5, 25, 50, 75, 95]))
