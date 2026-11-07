"""Shared loading: per-order S constraints for round 2."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from analysis.cache import load_tape  # noqa: E402

HERE = Path(__file__).resolve().parent
CACHE = HERE / '_cache'
CACHE.mkdir(exist_ok=True)


def orders(path='data/round2.csv') -> pd.DataFrame:
    """Inserts with is_buy and the raw constraint interval on S at insert time.

    maker bid p:  S >= p + shift ; maker ask p: S <= p - shift
    taker buy p:  S in [p - off - .5, p - off + .5]; sell: [p + off - .5, p + off + .5]
    """
    f = CACHE / (Path(path).stem + '_orders.parquet')
    if f.exists():
        return pd.read_parquet(f)
    t = load_tape(ROOT / path)
    o = t.orders[['t_us', 'order_id', 'side', 'price', 'volume', 'ioc', 'filled', 'end_t_us', 'end', 'row']].copy()
    o['is_buy'] = o.side == 'B'
    o.to_parquet(f)
    return o


def bounds(o: pd.DataFrame, shift=0.05, off=1.0):
    p = o.price.to_numpy().astype(float)
    b = o.is_buy.to_numpy()
    ioc = o.ioc.to_numpy()
    lo = np.full(len(o), -np.inf)
    hi = np.full(len(o), np.inf)
    m = ~ioc & b; lo[m] = p[m] + shift
    m = ~ioc & ~b; hi[m] = p[m] - shift
    m = ioc & b; lo[m] = p[m] - off - 0.5; hi[m] = p[m] - off + 0.5
    m = ioc & ~b; lo[m] = p[m] + off - 0.5; hi[m] = p[m] + off + 0.5
    return lo, hi
