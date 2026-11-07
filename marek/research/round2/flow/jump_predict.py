"""Can the shock jump direction be predicted from anything observable during the 3 s drought or before?"""
import numpy as np, pandas as pd
from pathlib import Path
D = Path(__file__).parent; C = D / 'cache'
r = pd.read_pickle(C / 'rows.pkl'); ev = pd.read_csv(D / 'shock_events.csv')
tt = r.t.to_numpy(); mid = ((r.bid_after + r.ask_after) / 2).ffill().to_numpy()
def at(x): return mid[np.clip(np.searchsorted(tt, x, 'right') - 1, 0, None)]
ins = r[r.op == 0]
rows = []
for o, e in zip(ev.onset, ev.end):
    jump = at(e + 0.5) - at(e - 0.1)
    d = ins[(ins.t > o + 0.1) & (ins.t < e - 0.1)]
    mk = d[~d.ioc]; tk = d[d.ioc]
    imb = mk.vol[mk.side == 0].sum() - mk.vol[mk.side == 1].sum()
    # maker centre drift during drought: mean bid px + mean ask px relative to pre-onset mid
    pre = at(o)
    ctr = (mk.px[mk.side == 0].max() + mk.px[mk.side == 1].min()) / 2 - pre if len(mk) else np.nan
    rows.append(dict(jump=jump, prior60=pre - at(o - 60), drought_mid=at(e - 0.1) - pre, mk_buy_share=(mk.side == 0).mean(), mk_vol_imb=imb,
                     tk_signed=(np.where(tk.side == 0, 1, -1) * tk.vol).sum(), envelope_ctr=ctr,
                     mk_late_ctr=np.nanmean(np.where(mk.side == 0, mk.px + 0.5, mk.px - 0.5)[mk.t.to_numpy() > e - 1.0]) - pre))
X = pd.DataFrame(rows)
print(X.round(2).to_string())
print('corr with jump:'); print(X.corr()['jump'].round(3).to_string())
print('sign agreement drought_mid vs jump (nonzero):', np.mean(np.sign(X.drought_mid[X.drought_mid != 0]) == np.sign(X.jump[X.drought_mid != 0])).round(2))
