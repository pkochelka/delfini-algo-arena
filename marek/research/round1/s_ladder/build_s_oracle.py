"""Exact tape S (jump times, S after each jump) -> data/.cache/round1_S.npz, for the s_oracle research flag."""
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'strategies' / 'peter'))
from analysis.cache import load_tape  # noqa: E402
from r1_model import order_table  # noqa: E402
o = order_table(load_tape(str(ROOT / 'data/round1.csv')))
f = o[o.ioc]
S = 10000 + 0.026 * np.cumsum(np.where(f.is_buy, 1.0, -1.0) * np.sqrt(f.filled.to_numpy()))
np.savez(ROOT / 'data/.cache/round1_S.npz', t=f.t_us.to_numpy(), S=S)
print(len(S), S[-1])
