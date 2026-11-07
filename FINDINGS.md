# FINDINGS: round 1 research

This file collects what we have learned and is meant for future agents and teammates. Read it before you research or change a strategy. Each section links to a detailed report under `research/round1/`, which holds the scripts, tables and caveats. Tape statistics are in `data_explorer/<tape>/README.md`.

Last updated 2026-11-07 (round 1). All numbers are round1 PnL unless stated otherwise. **Folds** means the 10 walk-forward folds `data/splits_round1/wf/wf01-10`, summed. The cross-fold sd is about 2.5k. Compare variants on their per-fold paired difference, which is much tighter.

## Scoreboard

| Strategy | Folds total | Full tape | Notes |
| --- | ---: | ---: | --- |
| `strategies/peter/ladder_mm.py` (committed HEAD, cap 100) | 425,118 | 425,639 | Baseline. `marek/impact_mm.py` with overlays off is identical. |
| `strategies/marek/sig_ladder.py` (defaults) | **473,066** | **473,597** (verified independently) | Depletion guard: +48k, wins 10/10 folds, paired t ≈ 10 |
| `strategies/marek/q_ladder2.py` (defaults) | 453,901 | — | 100 at the touch + 100 one tick behind, plus `improve_both` |
| `strategies/marek/q_ladder.py` with `levels 2, size_top 100, size_deep 100` | 450,608 | 454,798 | +25.5k, paired t 3.6, wins 9/10. Loses 33k on sample_v2 |
| `strategies/marek/robust_ladder.py` (defaults) | 416,746 | 425,613 | −2% on round1; turns losing stress tapes flat (see Robustness) |
| `strategies/marek/inv_mm.py` (defaults) | 425,118 | — | ladder plus inventory switches, all off by default |
| `marek/acf_momentum_mm.py` / `peter/arb_mm.py` | — | 96k / 50k | Not competitive |

**Warning:** the working-tree `strategies/peter/ladder_mm.py` was edited during this research and is not committed. At one point it had `cap: 1000`, which breaches the limit and gave about −10M on stress tapes and folds. Later it had `size_top: 400`. Always pin `cap=100` when you A/B, or use the frozen copies (`strategies/marek/q_ladder.py`, `research/round1/robustness/ref/ladder_mm_head.py`).

## What round1 is

- **Round1 is very friendly to a large passive maker.** The mid is a pure random walk (variance ratio ≈ 1.0 at every horizon) and the spread is 1 tick 93% of the time. The touch is deep (about 1,316 lots per side), and takers lose 0.31 ticks on average, permanently. The tape behaves the same from start to end. See `data_explorer/round1/README.md`.
- **Book imbalance predicts the mid 1 s ahead.** A side share of 0.04 means −0.38 ticks; 0.96 means +0.37.
- **Large aggressive orders have a delayed impact.** Roughly half of the impact arrives after our 5 ms latency, and it does not revert within 5 s. Trading on it does not pay (see the `impact_mm.py` docstring).
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

1. **Combine the gains.** Nobody has tested `sig_ladder`'s guard together with `q_ladder`'s 100/100 split or with `robust_ladder`'s guards. pre-join and `levels 2` both place size one tick behind, so they may overlap; expect less than +73k. Build one strategy with all of them behind flags, A/B on the folds, and check it on sample_v2, random_stress and the thin-book tape (`research/round1/robustness/run_matrix.py`, generated tapes in `data/stress/`).
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
| `strategies/peter/tune.py` | `load_folds` / `run_fold` and the Optuna search; `marek/sweep_impact.py` shows a minimal fold A/B |
