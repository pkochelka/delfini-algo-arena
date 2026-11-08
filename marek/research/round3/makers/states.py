"""Per-time state labels from the flow agent: regime from flow/regimes.csv (S-based calm/volatile segments) and shock
phase from flow/shock_events.csv: drought [onset, t_jump), post [t_jump, t_jump + 3 s).
state(t_us) -> array of 'calm' | 'vol' | 'drought' | 'post'."""
from pathlib import Path
import numpy as np, pandas as pd

FLOW = Path(__file__).resolve().parents[1] / 'flow'
REG = pd.read_csv(FLOW / 'regimes.csv')
SHOCK = pd.read_csv(FLOW / 'shock_events.csv')
ON = (SHOCK.onset.to_numpy() * 1e6).astype(np.int64)
JMP = (SHOCK.t_jump.to_numpy() * 1e6).astype(np.int64)


def is_vol(t_us):
    t = np.asarray(t_us) / 1e6
    k = np.clip(np.searchsorted(REG.start.to_numpy(), t, 'right') - 1, 0, len(REG) - 1)
    return (REG.regime.to_numpy()[k] == 'volatile')


def state(t_us):
    t = np.asarray(t_us)
    out = np.where(is_vol(t), 'vol', 'calm').astype(object)
    k = np.searchsorted(ON, t, 'right') - 1
    kk = np.maximum(k, 0)
    dr = (k >= 0) & (t < JMP[kk])
    po = (k >= 0) & (t >= JMP[kk]) & (t < JMP[kk] + 3_000_000)
    out[dr] = 'drought'; out[po] = 'post'
    return out
