"""Shared loaders for the bigmoves analysis: tape (replayed), S path, shocks."""
import json, pickle, pathlib, csv
import numpy as np, pandas as pd
ROOT = pathlib.Path(__file__).resolve().parents[5]
R3 = ROOT / 'marek/research/round3'; R2 = ROOT / 'marek/research/round2'
import sys; sys.path.insert(0, str(ROOT))
from analysis.cache import load_tape

def tape_path(name):
    if name == 'r3': return ROOT / 'data/round3.csv'
    if name == 'r2': return ROOT / 'data/round2.csv'
    if name.startswith('s'): return ROOT / f'data/synth/r3_v1_seed{name[1:]}.csv'
    if name.startswith('v'): return ROOT / f'data/synth/r3_v2_seed{name[1:]}.csv'
    if name.startswith('q'): return ROOT / f'data/synth/r2_v2_seed{name[1:]}.csv'
    raise ValueError(name)

def load(name):
    tp = load_tape(tape_path(name), verbose=False)
    o = tp.orders.copy(); tr = tp.trades
    o['t'] = o.t_us * 1e-6
    o['end_t'] = (o.end_t_us.fillna(1e18)) * 1e-6
    return tp, o, tr

def s_path(name):
    """(t, S) arrays: exact S per insert for r2/r3; for synth: None (use truth)."""
    if name == 'r3':
        d = pd.read_parquet(R3 / 's_dynamics/s_path.parquet')
    elif name == 'r2':
        d = pd.read_parquet(R2 / 's_dynamics/s_path.parquet')
    else:
        return None
    return d.t_us.values * 1e-6, d.s_hat.values

def shocks(name, S=None):
    """list of dict(onset, tj, J): tj = jump time (synth: onset+3.0 exact; tapes: first insert with the new S)."""
    if name in ('r3', 'r2'):
        f = (R3 if name == 'r3' else R2) / 'flow/shock_events.csv'
        ons = [float(r['onset']) for r in csv.DictReader(open(f))]
        t, s = S
        out = []
        for a in ons:
            j0 = np.searchsorted(t, a + 2.8); j1 = np.searchsorted(t, a + 3.4)
            k = np.argmax(np.abs(np.diff(s[j0:j1])))
            tj = t[j0 + k + 1]
            at = lambda x: s[np.clip(np.searchsorted(t, x, 'right') - 1, 0, len(t) - 1)]
            out.append(dict(onset=a, tj=tj, tprev=t[j0 + k], J=at(tj + 0.05) - at(t[j0 + k] - 0.05),
                            S_new=at(tj + 0.05), S_pre=at(t[j0 + k] - 0.05)))
        return out
    tr = json.load(open(tape_path(name).with_suffix('.truth.json')))
    return [dict(onset=a, tj=a + 3.0, J=J) for a, J in zip(tr['onsets'], tr['jumps'])]
