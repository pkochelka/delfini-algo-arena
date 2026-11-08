# Round 3 improvement brainstorm: shared brief

Goal: roughly **double** the round-3 PnL of `strategies/marek/round3/r3_best.py` (another team is known to be far ahead).
Each agent owns one angle (below) and writes `marek/research/round3/ideas/<angle>.md`. Do not edit `r3_best.py`;
prototype in copies `strategies/marek/round3/ideas/<angle>_*.py`, with each mechanism behind a PARAMS flag
(off = identical to r3_best).

## Read first (in this order)
1. `marek/research/round3/FINDINGS.md`: data findings for round 3 (sections A) and the strategy work so far.
2. `marek/research/round3/{flow,makers,novelty}/README.md`: the generator (ladder maker, transient impact, regimes, shocks).
   S fit: `marek/research/round3/s_dynamics/params_v3.json`; exact S per insert: `s_dynamics/s_path.parquet`
   (`uv run --with pyarrow python ...`). Per-order caches exist under each agent's `cache/` folder.
3. `marek/research/round2/round2_generator.md` and the memory notes on the round-2 sprint: lineage fvs_mm → ... →
   shockgen_best → r3_lad → r3_best. Round-2 dead ends: non-shock overshoot (no exit liquidity behind walls in r2),
   passive unwind, raising caps, passive quoting in volatile.
4. `strategies/marek/round3/r3_best.py` itself (≈1350 lines; FVMM = S filter + quoting, FVAShock = shock sniper + oversize).

## Where r3_best's money comes from (round 3 tape, 795k total, net of 17.7k breach penalty)
| Part | Time | PnL | per s |
| --- | ---: | ---: | ---: |
| calm | ≈ 3800 s | 239k | ≈ 63 |
| volatile | ≈ 1660 s | 422k | ≈ 254 |
| shock windows (onset −1 s … +20 s, 27 shocks) | ≈ 570 s | 135k | ≈ 5k per shock |

Synth means: v1 (6 seeds) 923k, v2 (4 seeds, held out) 862k. Round 2: 761k.
- Volatile PnL is all IOC sniping of stale quotes; passive quoting is off there (turning it on loses).
- Position room (±100) is the binding resource in round 2; the strategy is pinned much of volatile time.
- Simulator rules (`simulator/main.py`): latency 5 ms, MAX_POSITION 100, MAX_POTENTIAL_POSITION 1000 (resting + in-flight),
  **breach penalty 10,000 per second while |position| > 100**. So 10 ms over the limit costs 100. The shock sniper
  already uses this ("oversize", xs_cap 950, then IOCs back). In r3 the ladder adds 500–750 lots per level at 2–7 ticks
  behind the touch, i.e. exit liquidity that round 2 lacked.
- Public feed: 20 ticks of depth per side, every market event, with order ids and owner; trades per event.

## Tools
- `uv run python marek/research/round3/strategy/eval3.py FILE [FILE...] --tape T --tag TAG [--set k=v ...]`:
  T = `data/round3.csv` (default), `synth` (= `data/synth/r3_v1_seed1-6.csv`), or a comma list, e.g.
  `data/round3.csv,data/synth/r3_v1_seed1.csv`. Held-out family: `data/synth/r3_v2_seed11-14.csv`. Prints and appends rows
  (pnl, breach, calm/volatile/shock parts, 12×500 s windows, per-shock PnL) to `strategy/results.jsonl`.
  `strategy/summ.py TAGPREFIX --base TAG` summarises. One full-tape run ≈ 80–100 s per process.
- Synth generator: `marek/research/round3/strategy/r3_synth.py` (truth.json has shock onsets and regimes).
- **The machine has 10 cores shared by 5 agents: run at most 2 simulator processes at a time** (screen on round3 + one
  synth seed; run 6 synth seeds only for a finalist). Use tags prefixed with your angle name.
- Accept a change only if it is positive on round 3 AND on the synth mean; shock PnL is heavy-tailed (one shock at 4127 s
  flips ±17k on small changes), so judge shock changes on synth.

## Deliverable (`ideas/<angle>.md`, plain style, numbers over adjectives)
1. Ideas, ranked by expected gain, each with: mechanism, evidence from the tape (measured upper bound where possible),
   estimated PnL on round 3, cost/risk, and status (tested with numbers / prototyped / untested).
2. For the top 1–3 ideas: a prototype and eval3 numbers (round 3 + synth), or a clear reason why not.
3. Dead ends with numbers, so nobody retries them.
