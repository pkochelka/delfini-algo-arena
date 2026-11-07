# FINDINGS: round 1 research

This file collects what we have learned and is meant for future agents and teammates. Read it before you research or change a strategy. Each section links to a detailed report under `research/round1/`, which holds the scripts, tables and caveats. Tape statistics are in `data_explorer/<tape>/README.md`.

Last updated 2026-11-07 (round 1). All numbers are round1 PnL unless stated otherwise. **Folds** means the 10 walk-forward folds `data/splits_round1/wf/wf01-10`, summed. The cross-fold sd is about 2.5k. Compare variants on their per-fold paired difference, which is much tighter.

## Scoreboard

| Strategy | Folds total | Full tape | Notes |
| --- | ---: | ---: | --- |
| `strategies/peter/ladder_mm.py` (committed HEAD, cap 100) | 425,118 | 425,639 | Baseline. `marek/impact_mm.py` with overlays off is identical. |
| `strategies/marek/s_ladder.py` (defaults) | **718,108** | — | Hidden-price S ladder, +215k vs sig_ladder. See section 0 |
| `strategies/marek/sig_ladder.py` (defaults) | **489,114** | **489,589** | Depletion guard + 200 ms flow term: +64k, wins 10/10 folds |
| `sig_ladder.py` with `flow2_k 0` (previous default) | 473,066 | 473,597 (verified independently) | Depletion guard: +48k, wins 10/10 folds, paired t ≈ 10 |
| `strategies/marek/q_ladder2.py` (defaults) | 453,901 | — | 100 at the touch + 100 one tick behind, plus `improve_both` |
| `strategies/marek/q_ladder.py` with `levels 2, size_top 100, size_deep 100` | 450,608 | 454,798 | +25.5k, paired t 3.6, wins 9/10. Loses 33k on sample_v2 |
| `strategies/marek/robust_ladder.py` (defaults) | 416,746 | 425,613 | −2% on round1; turns losing stress tapes flat (see Robustness) |
| `strategies/marek/inv_mm.py` (defaults) | 425,118 | — | ladder plus inventory switches, all off by default |
| `marek/acf_momentum_mm.py` / `peter/arb_mm.py` | — | 96k / 50k | Not competitive |

**Warning:** the working-tree `strategies/peter/ladder_mm.py` was edited during this research and is not committed. At one point it had `cap: 1000`, which breaches the limit and gave about −10M on stress tapes and folds. Later it had `size_top: 400`. Always pin `cap=100` when you A/B, or use the frozen copies (`strategies/marek/q_ladder.py`, `research/round1/robustness/ref/ladder_mm_head.py`).

## 0. s_ladder: trade on the hidden price S (biggest gain, `marek/research/round1/s_ladder/`)

- **Live S:** S_raw += 0.026·√(taker's total fill) per market taker, corrected so every new market order at the touch has bid ≤ S−0.05 ≤ S+0.05 ≤ ask. The error sd is 0.12 tick against the tape's exact S. An oracle with the exact S (`s_oracle` flag, research only) gains just +22k, so the estimate is good enough.
- **Deltas on the folds:**

  | Step | Fold total | Δ |
  | --- | ---: | ---: |
  | `s_pull` 0.45 (pull the adding part when dist < th, pre-join behind) | 601.5k | +109k vs sig_ladder |
  | `s_pull` 0.35 + IOC snipe (`s_take` 0.1) + inside quotes on 2+ tick spreads (`s_improve` 0.4) | 694.9k | +32k (snipe) and +33k (inside quotes) |
  | `s_pull_flat` 0 (also pull the flattening part once S is through our price) | 700.6k | +5.6k |
  | `s_gen` 1, `s_pull` 0.3 (each side quotes at the best price with dist ≥ th) = defaults | 706.9k | |

  On 5 fresh synthetic generator tapes (`r1_synth.py --tag val`), the defaults beat sig_ladder by +101k (5/5, t 39).
- **Lost:** the sig_ladder guard on top (−6k), a higher pull threshold (0.5: −18k), hysteresis, keep_flat 0 (−30k), no pre-join (−74k), and an inventory-scaled threshold (`s_pull_pos` > 0).
- **Edge model:** a fill's edge ≈ dist − 0.026·√(taker volume before us). Most fills are at the front of the queue (`decompose.py`).
- **Optuna** (`tune_s.py`, 70 trials, `tune_results.jsonl`): 718.5k. The rounded values are now the defaults: `s_pull` .33, `s_pull_flat` .13, `s_pull_pos` −.03, `s_take` .035, `s_improve` .2, `s_hyst` .035, `s_maxdepth` 2, `s_gap` .095, `s_take_max` 100, `topup` 30.
  - Folds: +11.2k (8/10, t 1.4).
  - Synth: +1.9k (3/5, within noise), so most of the gain is fit to the folds.
- **Also lost:**
  - A queue-aware threshold `s_qk` (dist < th + k·c·√ahead): −3k to −58k.
  - A second order one tick behind the touch while the side is good (`s_deep`): −129k to −140k, because it takes room the touch order needs.
- The exact-S oracle is now a research subclass, `marek/research/round1/s_ladder/s_ladder_oracle.py`, so the submission file has no numpy, `hasattr` or file reads.
- **Open:**
  - Robustness on sample_v2 and the stress tapes (untested; it relies on the round1 generator).
  - A sandbox upload to confirm it is accepted.

## What round1 is

- **Round1 is very friendly to a large passive maker.** The mid is a pure random walk (variance ratio ≈ 1.0 at every horizon) and the spread is 1 tick 93% of the time. The touch is deep (about 1,316 lots per side), and takers lose 0.31 ticks on average, permanently. The tape behaves the same from start to end. See `data_explorer/round1/README.md`.
- **Book imbalance predicts the mid 1 s ahead.** A side share of 0.04 means −0.38 ticks; 0.96 means +0.37.
- **Large aggressive orders have a delayed impact.** Roughly half of the impact arrives after our 5 ms latency, and it does not revert within 5 s. Trading on it does not pay (see the `impact_mm.py` docstring).
- **Round1 is synthetic, and a hidden fair price S is the right model.** S = 10000 + 0.026 · Σ sign · √filled over taker orders, and nothing else moves it. The generator is in `strategies/peter/round1_generator.md`. The mid is S plus a gap that closes by itself: sd 0.41 tick, AR(1) 0.73 per 100 ms, half-life about 220 ms. Regressing the 1 s mid change on S − mid gives slope 1.03, so the gap closes completely.
- **S − mid beats every public feature at predicting the mid** (`strategies/peter/r1_structure.py`). Out-of-sample R², leave-one-split-out:

  | Horizon | Past returns | Imbalance | Maker insert aggressiveness | All public | S − mid | S − mid + public |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: |
  | 0.5 s | 0.00 | 0.11 | 0.18 | 0.21 | **0.32** | 0.32 |
  | 1 s | 0.00 | 0.06 | 0.13 | 0.14 | **0.21** | 0.21 |
  | 5 s | 0.00 | 0.01 | 0.02 | 0.02 | **0.04** | 0.04 |

  S − mid wins in every one of the 16 splits. Its worst split (0.29 at 0.5 s) is above the best split for all public features (0.23). Adding the public features on top of S − mid gains less than 0.005.
- **No periodicity, trend or volatility regime** (`r1_structure.py`, sections B and C):
  - Periodogram of 1 s counts and squared returns (Fisher's g): p ≥ 0.09.
  - Timestamp phase within 1 s / 0.1 s / 10 ms / 1 ms is uniform: KS p ≥ 0.26.
  - Return and taker-side autocorrelation is inside the noise band.
  - The trend in arrival rates is not significant (p ≥ 0.36).
  - S drifts +29.5 ticks over the tape (z = 0.29), and takers buy 50.02% of the time.
  - Squared returns at lag 300 give Ljung-Box p ≈ 0.02–0.05. That is borderline once you account for multiple testing, so there is no usable volatility clustering.
- **Every split looks the same.** These are the 12 walk-forward eval windows plus the 9000–12000 s holdout cut into 4.
  - Rates, sizes, lifetimes, spread, touch depth, gap sd and half-life, 1 s volatility and impact c (0.024–0.026) have a cross-split CV of 0.3–3.5%.
  - Homogeneity tests pass: taker counts p = 0.80, maker counts p = 0.46, buy share p = 0.35.
  - Per-split S drift is within |z| ≤ 1.34, which is noise. Fold-to-fold PnL differences are luck, not regime.
- **Anomaly: one side of the book empties for a few ms**, 140 times across the whole tape (the last at t = 11,919 s), not only at warm-up.
  - 131 episodes start with one large IOC clearing the whole side: median size 709 lots, against a median taker order of 61. The other 9 are cancels.
  - The side stays empty for a median of 9 ms and at most 83 ms.
  - **Consequence for code:** a mid built from raw `bid`/`ask` columns jumps by about 5,000 ticks, so skip or forward-fill one-sided moments. All current strategies (`ladder_mm`, `queue_mm`, `sig_ladder`, `q_ladder2`, `robust_ladder`) return early when either touch is `None`. They neither requote nor cancel then, which is safe. Keep that guard in new strategies.
  - **Not an edge:** with 4 takers/s per side and a window of about 9 ms (minus 5 ms latency), quoting into the empty side would catch roughly 5 taker orders on the whole tape.
- **Other ms-scale irregularities have mechanical causes.**
  - The busiest 100 ms window has 78 events against a mean of 37 (p ≈ 3e-4). This is most likely makers cancelling together after a jump in S.
  - 125 maker quotes sit more than 5 ticks from S, against about 36 expected from the offset distribution. The likely cause is the no-cross clip when the opposite touch has moved past S.
  - Maximum order sizes and maximum 10 s moves match the fitted distributions.
- **sample_v2** has mild momentum, a thinner book and wider spreads. **random_stress** and **sample** trend (variance ratio 1.6–2.4), and those are the tapes where the ladder loses. Expect the private and later tapes to look more like those than like round1.

## How ladder_mm makes money

Sources: `research/round1/inventory.md` and `research/round1/robustness.md`.

- **PnL breakdown:**
  - Spread capture is about +0.5 tick per lot. Inventory moves cost money, almost all of it within 10 s of the fill (adverse selection). End-of-tape liquidation costs about 50 per run, which is negligible.
  - The two agents measured different totals because they used different mid conventions. Inventory: spread +969k, inventory −543k. Robustness: spread +1.336M, inventory −911k.
- **The PnL does not come from the up-drift.** The average position is about 0, and the agents found it pinned at ±100 between 50% and 79% of the time, depending on how they measured. Pinned episodes are short (median 0.4 s), and long and short pins are symmetric. Folds where the mid rose made 210k; folds where it fell made 215k. **Mirrored-price folds give identical PnL.**
- **Pinning is the edge, not a leak.** Flow on the side the cap blocks is worthless (−0.03 tick per lot); flow on the side that reduces the position is worth +0.30. One big taker typically flips us from +100 to −100.
- **Queue position drives markout.** At 30 s, fills with 0 lots ahead mark out +0.20 to +0.22, while fills with 1,000–2,000 lots ahead mark out −1.55 to −1.65.
- **The most costly fills** come when our side holds under 10% of the touch while an old level is being eaten. These are 19% of lots, marking out −0.21 at 30 s, about −107k.

## Improvements that work

### 1. Depletion guard: `sig_ladder.py`, +48k (signals.md)

- **The trigger** is a side being thin and depleted. Our side's share of the touch is below `g_share` = 0.10, the touch quantity has fallen to `g_dep` = 0.5 of its peak since that price became the touch, and taker flow against the side over the last 1 s adds a term with `flow_k` = 0.10.
- **The reaction:**
  - **keep-flat:** `modify_order` the oldest touch order down to |pos|. The part that flattens the position keeps its queue priority; only the part that would flip the position is removed.
  - **pre-join:** rest the removed part one tick behind the touch. If the level breaks, that order is near the front of the new touch.
- **The depletion condition does the work.** A plain imbalance guard also fires at fresh levels, where the ladder earns most; that is why `imb_pull` failed. Without the depletion condition, the guard gives −7k.
- **Neighbouring parameters land at +29k to +36k for the guard alone** (before the flow term), so the result is a plateau rather than a lucky point. The flow term adds about +12k.
- **Porting caveat:** keep the `mod_n` bookkeeping for rejected or overlapping modifies. One buggy version lost track of an order and breached, scoring −10M.

### 1b. Short flow window in the guard score: `sig_ladder.py` `flow2_k 0.5, flow2_ms 200`, +16k (combo/)

- Subtracts `flow2_k` × (taker flow against the side over the last 200 ms) / 1000 from the score, on top of the 1 s term. Matches the OLS in signals.md, where 100 ms flow carried signal on its own.
- +16.0k on the folds (10/10, fold-delta sd 0.7k, t 6.9) and +16.0k on the full tape. It is a plateau: +12k to +16k over k 0.25–0.75 and windows of 200–500 ms. Dropping the 1 s term costs 5.5k, so both terms count. 50 ms is weaker (+4.9k), and k ≥ 2 loses.
- Stress tapes: +0.5k to +2.5k on toxic, drift, makers, takers and mirror, −1.6k on bigtaker4, −2.4k on sample_v2 (−0.9%), and flat on random_stress and sample.
- **Tested in the same sweep and left off:** the 100/100 touch/behind split (`size_deep`) does **not** stack with the guard (100/100 −6.5k, 150/50 −33k, 200/100 −81k), because pre-join already covers it. `improve_both` gives +3.3k on round1 (9/10) but −9k on pscale2 and −5k on random_stress, because its 3-tick threshold is absolute. A "severe" tier that also pulls the flattening part (`sev_share`) is noise.
- Script: `marek/research/round1/combo/sweep_combo.py`. Raw runs are in `results.jsonl` and the stress runs in `matrix.csv`. Note that `run_matrix.py`'s ROOT is one level off since the move to `marek/`, so pass it absolute paths.

### 2. Split between the touch and one tick behind: `q_ladder.py` `levels 2`, +25k (queue.md)

- **The problem it fixes:** when the touch breaks, the plain ladder lands at the back of the new touch (a median of 1,362 lots ahead), or cannot quote at all because the break pinned it.
- **With 100 at the touch and 100 one tick behind,** we already rest at the new touch in 90% of the breaks the side can quote, with a median 23 lots ahead. Fills at the front mark out +0.35 instead of +0.22.
- **The split is narrow:** 100/100 or `top_frac 0.5` gains +25k, 125/75 gains +11k, 150/50 loses 9k and 50/150 loses 49k.
- **It costs −33k on sample_v2,** where the touch itself is the edge. Re-validate it per tape.
- **`improve_both`:** step inside on both sides only when the spread is at least 3 ticks, worth +3.3k (t ≈ 4). It is in `q_ladder2.py`.

### 3. Robustness guards: `robust_ladder.py`, insurance (robustness.md)

- **Kill switch on our own mark-to-mid equity:**
  - It triggers on a 1,500 drop below the trailing 60 s peak, or a negative trailing 300 s PnL. The 1,500 scales with measured volatility relative to round1's 0.92.
  - It then caps inventory at 20 for 120 s. Repeat triggers double the cooldown and halve the cap.
- **Depth-scaled cap:** max(25, 0.1 × average touch size).
- **Shrinks orders with modify** when the cap drops, so they keep queue priority.

| Tape | ladder | robust_ladder |
| --- | ---: | ---: |
| random_stress | −18.6k | −0.1k |
| sample | −10.4k | −3.2k |
| thin book (×0.3) | 6.8k | 13.5k |
| sample_v2 | 263.5k | 251.3k |
| round1 folds | 425.1k | 416.7k |

**Why it is worth the 2% on round1:** the z-score counts a negative round as 0, so avoiding a losing round matters more than a small gain.

A lighter alternative from `inv_mm.py` (`adapt_w_s=300, adapt_thr=0, adapt_skew=50`) turns on skew only while the trailing 300 s PnL is negative. It changes nothing on round1 or sample_v2 and improves random_stress by +7.7k, but that comes from a single 2,000 s run.

## Rejected on round1 (don't redo without a new reason)

All tested on the round1 folds; numbers are deltas against ladder unless noted.

- **Inventory throttles:**

  | Variant | Δ |
  | --- | ---: |
  | `skew_lots` 50 / 25 | −14k / −46k |
  | `size_add` 50 / 25 | −5k / −14k |
  | `lim_add` | −12k to −26k |
  | `lim_flip` | −34k to −88k |
  | `add_back` | −1.7k |
  | passive unwind before the end | −1.6k to −44k |
  | `improve_reduce` | −134k |
  | `flat_at` | −485k to −972k |

  `flat_at` also **breaches the limit**: its IOC is outside the room guard. Never enable it unguarded. A guarded version exists in robust_ladder, off by default.
- **Stepping inside a 2-tick spread:** −13k to −153k for every side-selection rule. 2+-tick spreads last a median of 23 ms, and 70% close by continuation.
- **Cancel, freeze or shrink a whole side on imbalance alone:** −49k to −155k. Same for `imb_pull` and the "alone at the level" guard.
- **Sizing and placement:**
  - A fixed lower cap, `tight_offset` (−277k), `topup 0` (−67k), `size_top` below 200 with one level, 3 levels (−103k), `max_join`.
  - Moving size from the deep order to top up the touch: −133k. Holding the pre-joined order after release (`pj_hold`): −40k.
- **Overlays:**
  - Flow guards and IOC cover from `impact_mm`.
  - A variance-ratio guard, which adds nothing on top of the kill switch.
  - Volatility or deeper-book features for fill selection.
- **No effect (no need to re-test):** order chunking, modify vs cancel, and `max_spread` 3/8/20 (but `max_spread` 1 costs −45k).

## Next steps (open)

1. **Combine the gains.** Done for the split and `improve_both` (see 1b): neither stacks robustly. Still untested: sig_ladder plus `robust_ladder`'s kill switch. Original note: Nobody had tested `sig_ladder`'s guard together with `q_ladder`'s 100/100 split or with `robust_ladder`'s guards. pre-join and `levels 2` both place size one tick behind, so they may overlap; expect less than +73k. Build one strategy with all of them behind flags, A/B on the folds, and check it on sample_v2, random_stress and the thin-book tape (`research/round1/robustness/run_matrix.py`, generated tapes in `data/stress/`).
2. **Choose the round1 final.** One option is `sig_ladder` plus the robust guards, if the combination keeps most of the +48k. The 100/100 split is round1-specific (it loses on sample_v2), so include it only if the private tape is the same generator. Upload once to confirm the sandbox accepts it: these strategies import only the standard library and `simulator.models`.
3. **Commit the ladder_mm decision.** Settle the uncommitted `ladder_mm.py` edits (`cap`/`size_top`) and keep `cap ≤ 100` everywhere.
4. **Do this first in each new round:**
   - Export the tape (`uv run data-explorer --data data/roundN.csv --export`).
   - Run `research/round1/signals/race.py`, which checks that "thin and depleted means the level breaks" still holds.
   - Re-check the touch/behind split and re-tune `depth_k` and the kill threshold on the new folds.
   - Run the mirror and price-scale tapes as regression tests.
5. **Scale thresholds with the tape.** Every absolute $ or tick threshold should scale with volatility or spread; `max_spread: 8` already bites on the 4×-scaled tape.

## Where things are

| Path | Contents |
| --- | --- |
| `research/round1/inventory.md`, `inventory/` | PnL breakdown, pinning stats, inventory-variant sweeps (`harness.py`, `sweep_inv.py`) |
| `research/round1/signals.md`, `signals/` | Fill feature analysis (`build_fills.py`, `fill_2d.py`, `race.py`), sweep `sweep_sig.py`, raw `results.jsonl` |
| `research/round1/queue.md`, `queue/` | Level-transition instrumentation, spread episodes, A/B `ab.py` (pins cap=100) |
| `research/round1/robustness.md`, `robustness/` | Stress-tape generator `stress_tapes.py`, `run_matrix.py`, `decompose.py`, results CSVs |
| `strategies/marek/{sig_ladder,q_ladder,q_ladder2,robust_ladder,inv_mm}.py` | Variants; every new mechanic sits behind a PARAMS flag |
| `strategies/peter/round1_generator.md`, `r1_model.py`, `r1_synth.py` | The reverse-engineered generator: fit it on any tape, then generate synthetic tapes |
| `strategies/peter/r1_structure.py` | Structure audit: anomalies, periodicity, trends, per-split stability, predictive-model comparison. Run it first on each new round |
| `strategies/peter/tune.py` | `load_folds` / `run_fold` and the Optuna search; `marek/sweep_impact.py` shows a minimal fold A/B |
