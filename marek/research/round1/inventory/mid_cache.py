"""Extract a compact (t_us, bid, ask) top-of-book series from the round1 tape replay (market orders only).

    uv run python research/round1/inventory/mid_cache.py [tape]   # writes data/.cache/<tape>_quotes.npz
"""
import sys
from pathlib import Path

import numpy as np

from analysis import load_tape

ROOT = Path(__file__).resolve().parents[3]

if __name__ == '__main__':
    name = sys.argv[1] if len(sys.argv) > 1 else 'round1'
    tape = load_tape(ROOT / f'data/{name}.csv')
    q = tape.quotes
    q = q[q['bid'].notna() & q['ask'].notna()]
    np.savez(ROOT / f'data/.cache/{name}_quotes.npz', t_us=q['t_us'].to_numpy(np.int64),
             bid=q['bid'].to_numpy(np.float64), ask=q['ask'].to_numpy(np.float64))
    print(len(q), q['t_us'].iloc[0], q['t_us'].iloc[-1])
