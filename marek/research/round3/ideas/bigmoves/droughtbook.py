"""Resting market qty within 5 / 20 ticks of the touch at onset-0.5 and onset+2.95 (both sides), per tape."""
import sys, numpy as np, pandas as pd
from common import *
for name in sys.argv[1:]:
    tp, o, tr = load(name)
    S = s_path(name); sh = shocks(name, S)
    mk = o[~o.ioc & (o.resting_quantity > 0)]
    fl = tr.groupby('maker_row').apply(lambda g: (g.t_us.values, g.qty.values)) if False else None
    res = []
    for s in sh:
        for lab, x in (('pre', s['onset'] - 0.5), ('jump', s['onset'] + 2.95)):
            lv = mk[(mk.t < x) & (mk.end_t >= x)]
            # approx remaining: resting minus fills after insert is ignored -> upper bound; use only unfilled orders
            lv = lv[(lv.filled == lv.volume - lv.resting_quantity) | (lv.end_t >= x)]
            bb = lv[lv.side == 'B'].price.max(); ba = lv[lv.side == 'A'].price.min()
            b5 = lv[(lv.side == 'B') & (lv.price >= bb - 4)].resting_quantity.sum(); a5 = lv[(lv.side == 'A') & (lv.price <= ba + 4)].resting_quantity.sum()
            res.append(dict(lab=lab, spread=ba - bb, q5=(b5 + a5) / 2, n=len(lv)))
    R = pd.DataFrame(res)
    print(name, R.groupby('lab').median().round(1).to_dict('index'))
