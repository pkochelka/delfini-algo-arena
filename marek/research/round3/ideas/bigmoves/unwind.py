"""Per shock in a sim log: entry (jump direction) and exit (opposite) fills in [tj-0.3, tj+2], valued at S_new;
time spent over |100|; penalty."""
import sys, numpy as np, pandas as pd
from common import *
name, simlog = sys.argv[1], sys.argv[2]
S = s_path(name); sh = shocks(name, S)
tp, o, tr = load(name); ev = tp.events; evt = ev.t_us.values * 1e-6
st = pd.read_csv(pathlib.Path(simlog) / 'trades.csv')
st = st[(st.taker_owner == 'user') | (st.maker_owner == 'user')].copy(); st['t'] = st.timestamp_us * 1e-6
st['sgn'] = np.where(((st.taker_owner == 'user') & (st.taker_side == 'B')) | ((st.maker_owner == 'user') & (st.taker_side == 'A')), 1, -1)
pos = np.cumsum(st.sgn.values * st.quantity.values); tt = st.t.values
rows = []
for k, s in enumerate(sh):
    J = s['J']; d = 1 if J > 0 else -1
    if 'S_new' in s: Sn = s['S_new']
    else:
        i = np.searchsorted(evt, s['onset'] + 2.995) - 1; Sn = (ev.bid.iloc[i] + ev.ask.iloc[i]) / 2 + J
    a, b = np.searchsorted(tt, s['tj'] - 0.3), np.searchsorted(tt, s['tj'] + 2.0)
    m = st.iloc[a:b]; p0 = pos[a - 1] if a > 0 else 0
    en = m[m.sgn == d]; ex = m[m.sgn == -d]
    # time over 100 within window
    ps = np.r_[p0, pos[a:b]]; ts = np.r_[s['tj'] - 0.3, tt[a:b], s['tj'] + 2.0]
    over = sum(ts[i + 1] - ts[i] for i in range(len(ps)) if abs(ps[i]) > 100)
    rows.append(dict(k=k, J=round(J, 1), pos0=p0, maxpos=int(np.max(np.abs(ps))), q_in=en.quantity.sum(),
                     v_in=round(((Sn - en.price) * d * en.quantity).sum()), q_out=ex.quantity.sum(),
                     v_out=round(((ex.price - Sn) * d * ex.quantity).sum()), over_ms=round(over * 1e3), pen=round(over * 1e4)))
X = pd.DataFrame(rows); pd.set_option('display.width', 200)
print(X.to_string()); print(X.sum().round().to_string())
