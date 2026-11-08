# Round 4 generator reverse-engineering — shared brief

Tape: `data/round4.csv` (6000 s, 1,228,523 inserts, 58,445 IOC takers, 1,169,997 cancels). Columns as rounds 2–3.
Goal: write down the generator exactly, as for round 3 (`marek/research/round3/round3_generator.md`).
**Prior = round 3's model** (round 2 + transient impact + 2 ms ladder maker). Look for what changed or was added.

## Prior work to reuse (read before coding)
- Round 3 spec: `marek/research/round3/round3_generator.md`; details in `marek/research/round3/{s_dynamics,flow,makers,novelty}/README.md`.
- Round 3 code to copy/adapt (do NOT edit round 3 files): `marek/research/round3/s_dynamics/*.py` (decaying-impact S
  filter, `common.py`), `flow/*.py` (regimes, ladder_build, shocks), `makers/*.py`, `novelty/*.py` (battery of r2-vs-r3 tests).
- Round 2 spec: `marek/research/round2/round2_generator.md`. Compiled logs: `FINDINGS.md`, `marek/research/round3/FINDINGS.md`.
- Run with `uv run --with pyarrow --with numba python ...`. Don't edit pyproject.toml. Big intermediates go in a `cache/`
  or `_cache/` subfolder (git-ignored). Scripts write only inside your own folder.
- Lead recon scripts: `marek/research/round4/lead/recon*.py`.

## Lead recon (2026-11-08) — round 4 vs round 3
- **Unchanged-looking:** maker rate 195 /s (r3 196); taker size lognormal ≈ (4.12, 1.39) (r3 4.09, 1.40); side 50/50;
  the 2.000 ms ladder signature is still there (30.7k exact-2000 µs maker-insert gaps, r3 31.5k); cancel-life quantiles similar.
- **Shocks:** 28 droughts (<50 maker inserts per 0.5 s), onsets ≈ 167.5, 462, 661.5, 811.5, 1007, 1175, 1391, 1541, 1737.5,
  1916, 2168, 2355.5, 2538, 2686.5, 2940, 3143, 3384.5, 3565.5, 3833, 4076, 4259.5, 4467.5, 4711.5, 5008.5, 5243, 5422,
  5630, 5799. Crude jump sd ≈ 66 (r3 50), first onset earlier (167 s), some gaps < 150 s (150 s at 811.5→… check).
- **Much bigger price drift:** range 9178–10033 (r3 9758–10138). Mid-proxy 1-min change sd 49 (r3 28). Moves between
  shocks are **super-diffusive** in r4 (300 s sd 115 vs 73 from √-scaling of 1 s moves) where r3 was sub-diffusive (41 vs 49).
- **Taker sign is now persistent:** sign ACF lag 1–5 = 0.079, 0.072, 0.064, 0.057, 0.048 (r3 ≈ 0.00). corr(5 s taker
  imbalance, past 5 s mid move) 0.67 (r3 0.40); corr(imbalance, NEXT 5 s move) +0.135 (r3 0.00); at 20 s horizon −0.145.
  Candidates: meta-orders / an informed or trend-following taker class / a hidden drift that takers follow.
- **New reactive maker, size 15:** 382 maker inserts exactly **1 µs after an IOC taker** (r3: 13). 364 of them are exactly
  15 lots, 97 % on the side opposite to the taker, at the taker's limit price ±1 (e.g. buy taker limit 9999 → ask 15 @ 9999).
  Episodic (e.g. 20–27 s, 111–119 s: runs of buy takers while the price falls). Size 15 shows no spike in the overall maker
  histogram, so the agent may also post 15-lot orders not at 1 µs — check.
- **Cancel → insert 1 µs pairs:** 690 (r3 360); 73 % same side, 52 % same volume, price ±1–3 (cancel/replace = re-quote?).
- 1 µs Insert(F) → Insert(G) is the strongest new microstructure tell. Excess maker inserts 1–100 µs after a taker.

## Agents and ownership
- **s_dynamics/** — the hidden price S. Adapt round 3's decaying-impact filter; publish `s_dynamics/s_path.parquet` (one row
  per insert, same columns as round 3) ASAP and announce on the board. Then: impact law (c, exponent, p, T, noise), whether
  S has drift/trend/momentum of its own (the super-diffusion), regime switch times, shock jumps.
- **flow/** — takers and arrivals: regimes (HMM), rates, shock schedule/drought/jump law, and above all **the taker sign
  persistence**: is it a separate class (meta-orders with a hidden direction, informed takers who know future S / drift,
  trend followers who chase past S moves)? Fit the mechanism with numbers (class rate, run lengths, sizes, limits).
- **makers/** — base makers (offset law, cancels, sizes), the ladder (still the same?), and the **new 15-lot reactive maker**
  (trigger rule, price rule, side, cancels/lifetime, whether it posts outside the 1 µs pattern, what fraction of takers trigger
  it and why), plus the cancel→insert 1 µs pairs.
- **novelty/** — rerun round 3's battery (`marek/research/round3/novelty/`) on round 4 vs round 3 and rank every material
  difference; follow up whatever the other three don't own. Cross-check the others' claims.

## Collaboration
- **Shared board: `marek/research/round4/NOTES.md`.** Append short dated, signed entries (`[agent] finding`) whenever you
  learn something another agent can use. Re-read the board every ~20–30 min of work and before writing your final README;
  adopt others' results instead of re-deriving them. Append with `cat >> ... <<'EOF'` (never overwrite the board).
- Until `s_path.parquet` exists use a proxy (maker-insert bounds / mid, or round 3's proxy code).
- Finish with a `README.md` in your folder: identified model with numbers, confidence, validation, what's still open, and
  **trading implications** (what a market maker can exploit: predictability of S, toxic flow, tells). Plain style as round 3.
