"""Orders (prep.load) + ladder chain ids (cache/chains.pkl) + S per insert from s_dynamics/s_path.parquet + state."""
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
    o['lad'] = (o.L >= 2) & (o.L <= 6) & ~o.ioc.astype(bool)
    o['ioc'] = o.ioc.astype(bool)
    return o
