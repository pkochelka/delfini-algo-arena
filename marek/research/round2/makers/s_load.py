"""S per insert (smoothed posterior mean, from s_dynamics agent's grid HMM, smooth_v1) aligned with prep.load()."""
from pathlib import Path
import numpy as np
from prep import load

HERE = Path(__file__).parent


def with_s(path=None):
    o = load()
    p = Path(path) if path else HERE / 'cache/s_smooth_v1.npz'
    z = np.load(p)
    o['S'] = z['mean']; o['S_sd'] = z['sd']; o['S_lo'] = z['lo']; o['S_hi'] = z['hi']
    return o
