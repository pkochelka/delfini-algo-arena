# Round 3 generator reverse-engineering — shared brief

Tape: `data/round3.csv` (6000 s, 1,234,596 inserts, 56,274 IOC takers, 1,178,209 cancels). Columns as round 2.
Goal: write down the generator exactly as we did for round 2 (`marek/research/round2/round2_generator.md`).
Round 3 is "harder" than round 2. Assume round 2's model is the prior and look for what changed or was added.

## Prior work to reuse (read before coding)
- Round 2 spec: `marek/research/round2/round2_generator.md`, details in `marek/research/round2/{s_dynamics,flow,makers}/README.md`.
- Round 2 code to copy/adapt (do not edit round 2 files): `marek/research/round2/s_dynamics/*.py` (HMM S reconstruction,
  `common.py` loads per-order constraints via `analysis.cache.load_tape`), `marek/research/round2/flow/*.py`, `makers/*.py`.
- Round 1 generator: `strategies/peter/round1_generator.md`. Compiled log: `FINDINGS.md`.
- Venv lacks numba/pyarrow: run with `uv run --with pyarrow --with numba python ...`. Don't edit pyproject.toml.
- Big intermediates go in a `_cache/` or `cache/` subfolder (git-ignored pattern used in round 2).

## First recon (lead, 2026-11-08)
- Same skeleton as round 2: taker sizes lognormal(4.09, 1.40) unchanged; taker sign ACF ≈ 0; side 50/50.
- 27 maker droughts (<50 maker inserts per 0.5 s), onsets ≈ 254.5, 471, 663.5, 956, 1169, 1356.5, 1533, 1779, 1981,
  2206, 2375.5, 2614, 2896.5, 3042, 3187.5, 3404.5, 3669, 3902.5, 4127, 4298.5, 4559, 4806, 5034, 5243.5, 5461, 5701.5, 5955.
- **Changed:** maker rate 196 /s (r2 178); per-minute maker count sd 745 (r2 199). Some minutes have +15–20 % maker
  inserts AND larger median maker size (16 → 21–24), clustered in multi-minute runs (e.g. min 10–13, 38–41, 63–68,
  92–98). Maker log-size mean 2.83 vs 2.71. Candidate new mechanism (a third regime? a new participant class?).
- Price range 9732–10160 (r2 9444–10017): smaller total drift.

## Collaboration
- Each agent owns one folder: `s_dynamics/`, `flow/`, `makers/`, `novelty/`. Write only there (plus the board).
- **Shared board: `marek/research/round3/NOTES.md`.** Append short dated, signed entries (`[agent] finding`) whenever you
  learn something another agent can use (e.g. regime switch times, new participant signature, S path ready).
  Re-read the board every ~20–30 min of work and before writing your final README; adopt others' results instead of
  re-deriving them. Append with `cat >> ... <<'EOF'` (never overwrite the board).
- The s_dynamics agent publishes `s_dynamics/s_path.parquet` (one row per insert, same columns as round 2's) as soon as a
  usable version exists and announces it on the board. Until then use a proxy (e.g. maker-insert bounds / mid).
- Finish with a `README.md` in your folder: identified model with numbers, confidence, validation, what's still open,
  trading implications. Write in the plain style of the round 2 READMEs.
