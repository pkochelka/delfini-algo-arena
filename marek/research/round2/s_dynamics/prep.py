"""Arrays for the HMM: per insert time, kind, price, window centre, filled, sign."""
from __future__ import annotations

import numpy as np
import pandas as pd

import common
from common import load_tape

H_STEP = 0.05


def build(path='data/round2.csv', h=H_STEP):
    f = common.CACHE / (path.split('/')[-1].replace('.csv', '') + f'_hmm_{h}.npz')
    if f.exists():
        return dict(np.load(f))
    T = load_tape(common.ROOT / path)
    o = T.orders
    ev = T.events
    prev = np.maximum(o.row.to_numpy() - 1, 0)
    bb = ev.bid.to_numpy()[prev].astype(float)
    ba = ev.ask.to_numpy()[prev].astype(float)
    bb[(bb == 0) | (o.row.to_numpy() == 0)] = np.nan
    ba[(ba == 0) | (o.row.to_numpy() == 0)] = np.nan
    p = o.price.to_numpy().astype(float)
    buy = (o.side == 'B').to_numpy()
    ioc = o.ioc.to_numpy()
    kind = np.full(len(o), 6, np.int64)
    kind[~ioc & buy] = 0
    kind[~ioc & ~buy] = 1
    kind[~ioc & buy & (p == ba - 1)] = 2
    kind[~ioc & ~buy & (p == bb + 1)] = 3
    kind[ioc & buy] = 4
    kind[ioc & ~buy] = 5
    # robust centre: centred rolling median of maker prices (+-15 makers), ffilled to all inserts
    mp = pd.Series(np.where(~ioc, p + np.where(buy, 0.5, -0.5), np.nan))
    mk = mp.dropna()
    med = mk.rolling(31, center=True, min_periods=1).median()
    cen_px = med.reindex(mp.index).ffill().bfill().to_numpy()
    cen = np.round(cen_px / h).astype(np.int64)
    t = (o.t_us.to_numpy() - T.start_us) / 1e6
    out = dict(t=t, t_us=o.t_us.to_numpy(), kind=kind, price=p, cen=cen, filled=o.filled.to_numpy().astype(float),
               volume=o.volume.to_numpy().astype(float), sign=np.where(buy, 1.0, -1.0), ioc=ioc, bb=bb, ba=ba)
    np.savez(f, **out)
    return out
