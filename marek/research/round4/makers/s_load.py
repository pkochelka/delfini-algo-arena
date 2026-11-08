"""Orders (prep.load) + ladder chains (cache/chains.pkl) + S v0 per insert + state + class labels (cache/order_class.parquet).
lad = any non-base maker (ladder chain or r15 agent), so 'not lad and not ioc' = base makers."""
from pathlib import Path
import numpy as np, pandas as pd
import states
HERE = Path(__file__).parent


def with_s(path=None):
    o = pd.read_pickle(HERE / 'cache/chains.pkl')
    s = pd.read_parquet(path or HERE.parent / 's_dynamics/s_path.parquet')
    assert len(s) == len(o) and (s.t_us.to_numpy() == o.t_us.to_numpy()).all()
    o['S'] = s.s_hat.to_numpy(); o['S_sd'] = s.s_sd.to_numpy(); o['S_lo'] = s.s_lo.to_numpy(); o['S_hi'] = s.s_hi.to_numpy()
    o['st'] = states.state(o.t_us.to_numpy())
    c = pd.read_parquet(HERE / 'cache/order_class.parquet').cls.to_numpy()
    o['cls'] = c
    o['lad'] = (c == 1) | (c == 2)
    o['ioc'] = o.ioc.astype(bool)
    return o
