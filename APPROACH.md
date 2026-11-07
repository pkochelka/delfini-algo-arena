# APPROACH: how to attack a round

This is the playbook distilled from round 1. Round-specific results live in `FINDINGS.md`; this file is about method.

## The lesson from round 1

We had reverse-engineered the generator early: the hidden fair price S and the slow makers are in `strategies/peter/round1_generator.md`. Even so, we spent most of the round tuning proxies for S: imbalance, depletion and flow guards, inventory throttles, Avellaneda-Stoikov overlays. That got the ladder from 425k to 489k. A strategy built directly on S (`strategies/marek/s_ladder.py`) reached 707k within a few hours.

**Find the hidden state that drives the tape, measure what it is worth on our own fills, then build the strategy around it. Tune proxies only when no such state exists.**

## 1. First hours: understand the generator (before any strategy work)

1. **Export the tape.** Run `uv run data-explorer --data data/roundN.csv --export` and read `data_explorer/roundN/README.md`.
2. **Fit the round 1 model** with `strategies/peter/r1_model.py data/roundN.csv`. It reports how many orders break the price rules.
   - **0 violations:** the same structure holds. Re-use `s_ladder` and retune it.
   - **Many violations:** find what changed. Typical suspects:
     - S diffuses without trades (`s_sigma`).
     - Impact has a different exponent or decays.
     - Takers are informed.
     - Several fair prices.
     - Volatility regimes.
3. **Look for the latent variable** with the tests in `strategies/peter/r1_structure.py`:
   - Can a hidden price be reconstructed exactly from public events (trades, `taker_order_id`, insert prices)?
   - Do maker quotes bracket it?
   - Does mid − latent mean-revert?
   - Is its out-of-sample R² above the public features?
4. **Check who reacts how fast.** Measure the other makers' cancel delay after a move against them. Our 5 ms latency against their 20–40 ms is the edge we earn by pulling and sniping.
5. **Keep the generator up to date.** Once the model fits, regenerate tapes with `r1_synth.py` (update it if the process changed). They are the validation set.

## 2. Estimate the ceiling before building

Run a baseline (the ladder) and break its fills down by the latent state just before each fill. In round 1, the state was the distance of our price from S. Script: `marek/research/round1/s_ladder/decompose.py`.

- **The money map:** edge per lot by bucket, and total lots per bucket. Losing buckets are what pulling saves; winning buckets show where more volume pays.
- **The estimation ceiling:** an oracle run with the exact latent value (`s_ladder_oracle.py`). If the oracle is not much better, estimation is good enough and the gains are in the quoting logic.
- **The edge decomposition:** for a passive fill, edge ≈ distance − impact of the taker that hit us. That says which fills to avoid.

## 3. Build: a market maker driven by the latent state

The round 1 design, in order of value. Expect the same skeleton to transfer.

1. **Queue-priority ladder core:** one order per side at the touch, never reposted needlessly, and worst-case |position + open orders| ≤ cap.
2. **Pull by state:** a side is bad when our price is within the threshold of fair value.
   - Keep the part that flattens our position, using `modify_order` so it keeps priority.
   - Rest the rest one tick behind the touch, so we are early in the next queue.
3. **Snipe stale quotes:** send an IOC into quotes that are through fair value by a small margin, before the slow makers cancel. Limit the size to the cap room.
4. **Improve only when safe:** quote inside a wide spread only if the new price stays at the threshold from fair value.
5. **Leave out what loses** unless there is a new reason to try it:
   - Active unwinding.
   - Inventory skew or throttles, when the latent value has no drift.
   - Pulling whole sides.
   - Holding orders after the state clears.

## 4. Evaluate honestly

- **Paired per-fold deltas** on the walk-forward folds (`marek/research/round1/s_ladder/sweep_s.py`, about 20 s for 10 folds). Report the total, the delta, its sd, t and wins out of 10. Fold-to-fold PnL differences are mostly luck; the paired delta is the signal.
- **Synthetic holdout:** `sweep_s.py --synth` runs on fresh generator seeds. The private tape is most likely another draw of the same process, so this is the closest stand-in for the leaderboard.
- **Stress tapes** (`research/round1/robustness/`, `data/stress/`, sample_v2, random_stress) to check that a mechanic does not blow up when the structure shifts.
- **Tune last,** with Optuna over a few meaningful parameters (`tune_s.py`). Confirm the winner on the synth tapes, not only on the folds it was tuned on.

## 5. Score-aware choices

- The score is the sum of z-scores per round, weighted 0.8 / 1.0 / 1.0 / 1.2, and a negative round counts as 0. Avoiding a loss matters more than a few % of upside.
- Later rounds will be harder (sample, random_stress and sample_v2 trend). Keep a regime gate ready, such as robust_ladder's kill switch, and only enable it when the round's tape shows the need.
- Scale every absolute threshold with measured volatility, spread and tick size.

## 6. Engineering rules

- **One mechanic per PARAMS flag,** with defaults that reproduce the previous best bit for bit. Never delete a mechanic; turn it off.
- **Record every result:** each sweep appends to `results.jsonl`, and wins and losses go into `FINDINGS.md`.
- **Commit at least every 30 minutes,** including failed experiments.
- **Submission hygiene:**
  - Only the standard library and `simulator.models` / `BaseStrategy`. No `getattr`, `type`, `open`, `os`, `sys` or numpy.
  - Research-only code (oracles, file reads) belongs in subclasses under `marek/research/`.
  - Upload once early to confirm the sandbox accepts the file.
- **Position safety:**
  - Always pin `cap ≤ 100`. The breach penalty is 10,000 per second.
  - Keep the `mod_n` bookkeeping for modifies that are rejected or overlap. A version that lost track of an order scored −10M.
- **Don't edit a strategy file while a sweep or tune is running.** Every trial reloads it from disk.
- **Return early when either side of the book is empty.** One side can be empty for a few ms, and a mid built from raw touches then jumps by thousands of ticks.

## 7. Round checklist

1. Export the tape, fit the generator and count the violations.
2. Find the latent state, measure the other makers' reaction delay, and update the generator.
3. Run the baseline, decompose its fills by state, and run the oracle ceiling.
4. Build or retune the state-driven market maker (section 3), adding one flag at a time with paired fold deltas.
5. Validate on synth seeds and stress tapes, then tune and re-validate.
6. Upload, mark the final submission, and update `FINDINGS.md` and this file with what changed.
