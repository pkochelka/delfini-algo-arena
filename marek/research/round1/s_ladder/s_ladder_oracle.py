"""Research only: s_ladder with the tape's exact S (data/.cache/round1_S.npz from build_s_oracle.py)."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'strategies' / 'marek'))
import s_ladder  # noqa: E402

PARAMS = s_ladder.PARAMS
Z = np.load(ROOT / 'data/.cache/round1_S.npz')


class SLadderOracle(s_ladder.SLadder):
    def __init__(self) -> None:
        super().__init__()
        self._ot, self._os, self._oi = Z['t'].tolist(), Z['S'].tolist(), 0

    def _s_value(self, now: int) -> float:
        t, i = self._ot, self._oi
        while i < len(t) and t[i] <= now:
            i += 1
        self._oi = i
        return self._os[i - 1] if i else 10000.0
