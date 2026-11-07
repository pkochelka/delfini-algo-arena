"""Per-time state labels from the flow agent: regime (regimes_envW.csv: calm/volatile segments, 1 s resolution)
and shock phase (shock_events.csv onsets: drought [onset, onset+3 s), post [onset+3, onset+6 s)).
state(t_us) -> array of 'calm' | 'vol' | 'drought' | 'post'."""
from pathlib import Path
import numpy as np, pandas as pd

FLOW = Path(__file__).resolve().parents[1] / 'flow'
REG = pd.read_csv(FLOW / 'regimes_envW.csv')
SHOCK = pd.read_csv(FLOW / 'shock_events.csv')
ONSETS_US = (SHOCK.onset.to_numpy() * 1e6).astype(np.int64)


def is_vol(t_us):
    t = np.asarray(t_us) / 1e6
    k = np.clip(np.searchsorted(REG.start.to_numpy(), t, 'right') - 1, 0, len(REG) - 1)
    return REG.high.to_numpy()[k].astype(bool)


def shock_phase(t_us):
    """seconds since the latest onset (np.inf if none)."""
    t = np.asarray(t_us)
    k = np.searchsorted(ONSETS_US, t, 'right') - 1
    ds = np.where(k >= 0, (t - ONSETS_US[np.maximum(k, 0)]) / 1e6, np.inf)
    return ds


def state(t_us):
    ds = shock_phase(t_us)
    v = is_vol(t_us)
    out = np.where(v, 'vol', 'calm').astype(object)
    out[(ds >= 0) & (ds < 3.0)] = 'drought'
    out[(ds >= 3.0) & (ds < 6.0)] = 'post'
    return out
