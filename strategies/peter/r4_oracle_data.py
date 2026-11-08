"""Build logs/r4/oracle_truth.npz for r4_oracle.py from Marek's round-4 s_dynamics outputs.

    uv run --with pyarrow python strategies/peter/r4_oracle_data.py
Needs marek/research/round4/s_dynamics/s_path.parquet (rebuild: excl.py, full_v2.py v0b, kfd.py v0b D,
full_v2.py v2b with VELFILE, kfd.py v2b D, full_v2.py v3 with VELFILE, export_path.py v3).
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
R4 = ROOT / 'marek/research/round4'
sp = pd.read_parquet(R4 / 's_dynamics/s_path.parquet', columns=['t_us', 's_hat', 's_filt'])
vs = pd.read_csv(R4 / 's_dynamics/vol_segments.csv')
sh = pd.read_csv(R4 / 'flow/shock_events.csv')
out = ROOT / 'logs/r4/oracle_truth.npz'
out.parent.mkdir(parents=True, exist_ok=True)
np.savez(out, t=sp.t_us.to_numpy() * 1e-6, S=sp.s_hat.to_numpy(), Sf=sp.s_filt.to_numpy(),
         va=vs.start.to_numpy(), vb=vs.end.to_numpy(),
         on=sh.onset.to_numpy(), tj=sh.t_jump.to_numpy(), J=sh.jump.to_numpy())
print(out, len(sp), 'inserts;', len(vs), 'vol segments;', len(sh), 'shocks')
