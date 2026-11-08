"""Lifetimes (insert -> cancel row) and fill shares by participant and state. Base = not chain ladder and not calm
ladder-like singleton (L==1, calm, 70-330 lots, placed >= 2 behind the touch)."""
import numpy as np, pandas as pd
from s_load import with_s
o = with_s(); m = o[~o.ioc].copy()
rel = np.where(m.sg > 0, m.price - m.bid, m.ask - m.price)
m['who'] = np.where(m.lad, 'ladder(chain)', np.where((m.L == 1) & (m.st == 'calm') & m.volume.between(70, 330) & (rel <= -2), 'ladder-like single (calm)', 'base'))
m['life'] = (m.cancel_us - m.t_us) / 1e6
rows = []
for (w, s), g in m.groupby(['who', 'st']):
    l = g.life.dropna()
    rows.append(dict(who=w, state=s, n=len(g), cancel_row_share=g.cancel_us.notna().mean(), filled_share=(g.filled > 0).mean(),
                     mean=l.mean(), p10=l.quantile(.1), p50=l.median(), p90=l.quantile(.9), p99=l.quantile(.99)))
df = pd.DataFrame(rows).round(3); pd.set_option('display.width', 200); print(df.to_string()); df.to_csv('lifetimes.csv', index=False)
b = m[(m.who == 'base') & (m.st == 'calm')]
print('base calm life by size band', b.groupby(pd.cut(b.volume, [0, 10, 100, 1000, 1e6])).life.mean().round(3).to_dict())
