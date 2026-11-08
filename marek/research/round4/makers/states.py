"""Per-time state labels for round 4: calm / vol from s_dynamics' S v0 per-insert vol flag (s_path.parquet), and shock
phase from flow/shock_events.csv: drought [onset, t_jump), post [t_jump, t_jump + 3 s). state(t_us) -> array of str."""
from pathlib import Path
import numpy as np, pandas as pd
R4 = Path(__file__).resolve().parents[1]
_S = pd.read_parquet(R4 / 's_dynamics/s_path.parquet', columns=['t_us', 'vol'])
_T = _S.t_us.to_numpy(); _V = _S.vol.to_numpy()
SHOCK = pd.read_csv(R4 / 'flow/shock_events.csv')
ON = (SHOCK.onset.to_numpy() * 1e6).astype(np.int64)
JMP = (SHOCK.t_jump.to_numpy() * 1e6).astype(np.int64)


def is_vol(t_us):
    k = np.clip(np.searchsorted(_T, np.asarray(t_us), 'right') - 1, 0, len(_T) - 1)
    return _V[k]


def state(t_us):
    t = np.asarray(t_us)
    out = np.where(is_vol(t), 'vol', 'calm').astype(object)
    k = np.searchsorted(ON, t, 'right') - 1
    kk = np.maximum(k, 0)
    dr = (k >= 0) & (t < JMP[kk])
    po = (k >= 0) & (t >= JMP[kk]) & (t < JMP[kk] + 3_000_000)
    out[dr] = 'drought'; out[po] = 'post'
    return out
