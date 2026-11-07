# Round 1: queue mechanics, level transitions, spread regimes

Baseline: committed `strategies/peter/ladder_mm.py` (cap 100, size_top 200, levels 1, topup 10). On the 10
walk-forward folds (`data/splits_round1/wf`) it makes **425,118**. The full tape gives 425,639.

Noise yardstick: every A/B runs on the same 10 folds, so the line that matters is the **paired** fold
difference (`dSd` = sd of per-fold deltas, `t` = mean/(dSd/√10)). The cross-fold PnL sd is about 2.5k
per fold. A variant whose total is within ±5k of base and has |t| < 2 counts as noise.

> **Warning:** during this work someone changed `strategies/peter/ladder_mm.py` on disk to `cap: 1000`
> (uncommitted). Any run that loads it without pinning `cap` will breach and lose about −10M per
> fold. The A/B here uses `strategies/marek/q_ladder.py`, a frozen copy that matches the committed
> ladder_mm exactly (same per-fold PnLs). `ab.py` also pins `cap=100` on every variant. One early
> sweep was contaminated and has been moved to `queue/out/ab_results_contaminated.jsonl`; ignore it.

## Tools (research/round1/queue/)

- `ab.py`: paired A/B on the 10 folds with at most 3 workers. It appends per-fold results to `ab_results.jsonl`.
- `instrument.py`: wraps a strategy instance without changing its file. On every snapshot it records
  how many market lots sit ahead of each of our orders, and logs each change of the market touch: when we
  got in at the new touch and how deep in the queue. Output goes to `out/*.pkl`.
- `analyze.py`: fill markouts by queue position and placement, plus statistics on touch transitions.
- `spread_episodes.py`: episodes where the spread is 2 ticks or wider (duration, how they close, drift after).
- `strategies/marek/q_ladder.py`: ladder_mm plus switches (`improve` 1–6, `improve_both`, `keep_behind`,
  `deep_yield` modify/cancel, `chunk`, `pre_thin`/`pre_take`, `top_frac`/`top_min`). All are off by default.
- `strategies/marek/q_ladder2.py`: the same code with the winners as defaults (levels 2, 100 at the touch +
  100 one tick behind, `improve_both` 1).

## 1. Queue priority economics and level transitions (baseline instrumented, 10 folds)

- Our order activates a median **6.9 ms** after it is sent (p90 11.7 ms, p99 19.6 ms).
- Markout depends heavily on how many market lots were ahead of our order on the snapshot just before
  the fill (lot-weighted, ticks, our direction):

  | lots ahead | lots filled | 1 s | 30 s |
  |---|---|---|---|
  | 0 | 1.38M | +0.21 | +0.22 |
  | 1–200 | 1.11M | +0.14 | +0.14 |
  | 200–500 | 109k | −0.04 | −0.07 |
  | 500–1000 | 27k | −0.22 | −0.41 |
  | 1000–2000 | 7k | −0.43 | −1.55 |

- About 27.4k changes of the market touch price:
  - 14.0k are improvements, where a new level forms inside the spread. On a side that can quote, we join
    in a median 14 ms (p90 21 ms) with about 31 lots ahead. The new level is small (median 16 lots).
  - 13.4k are breaks: the level is consumed (9.6k by trades) or cancelled away (3.8k).
- **Breaks are where the baseline loses priority.** When the side can still quote (3.3k breaks), we
  reach the new touch after a median 14.6 ms, behind a median **1,362 lots**. That is the back of the
  thick second level, the queue position whose fills mark out badly. In 10.1k of the 13.4k breaks
  the side cannot quote at all, because the break filled us up to the ±100 cap. The baseline sits at
  |pos| = 100 in 79% of transitions.
- **Pre-positioning one tick behind fixes this.** With 100 lots at the touch and 100 one tick behind
  (levels 2), we are already resting at the new touch in 90% of quotable breaks, with a median of
  **23 lots ahead** instead of 1,362.
  - Fills get better: the ahead = 0 bucket marks out at +0.31/+0.35 against +0.21/+0.22 in the baseline.
  - Volume falls: 1.77M lots traded against 2.65M.
  - Inventory sits at the cap less: |pos| = 100 in 44% of transitions, against 79%.

### Variants (10 folds, total PnL, paired delta against base 425,118)

| variant | total | Δ | t | wins | note |
|---|---|---|---|---|---|
| **levels 2, top 100 / deep 100 (B2)** | **450,608** | **+25.5k** | +3.6 | 9/10 | full tape: **454,798** (+29.2k), pnl_se 9.3k vs 11.3k |
| levels 2, top 100 / deep 200 | 450,440 | +25.3k | +3.5 | 9/10 | deep is capped by the guard, so this matches B2 |
| levels 2, `top_frac` 0.5, `top_min` 0 (half of allowance at touch, half behind) | 450,453 | +25.3k | | | the same plateau with no tuned constant |
| top_frac 0.5 / top_min 75 | 453,353 | +28.2k | | | within noise of B2 (+2.7k, t 1.5) |
| levels 2, top 110 / 90, 90 / 110, 125 / 75 | 445.8k / 442.7k / 436.1k | +20.6 / +17.6 / +11.0k | | | the peak sits near an even split |
| levels 2, top 150 / 50 | 415,787 | −9.3k | −1.2 | | |
| levels 2, top 75 / 125, 50 / 150 | 424.5k / 376.3k | −0.6 / −48.8k | | | too little at the touch |
| levels 2, top 200 / deep 50–100 | 354,002 | −71.1k | −10.6 | 0/10 | see below |
| levels 3, 100 / 50 / 50 | 321,820 | −103.3k | −14.6 | 0/10 | |
| `keep_behind` 1 (keep the old order when the touch improves) | 353,912 | −71.2k | −10.6 | 0/10 | |
| keep_behind 1 + `deep_yield` (modify or cancel) | 425,118 | 0 | | | with size_top 200 the yield always frees everything, so this is the baseline |
| B2 + deep_yield 1 (modify) / 2 (cancel) | 317.9k / 316.2k | −132.7 / −134.4k | | | moving weight back to the touch kills the gain |
| B2 + keep_behind 2 | 311,685 | −138.9k | | | |
| `pre_thin` 300 (deep only when the touch is thin) | 318,735 | −106.4k | | | |
| B2 with topup 0 / 5 / 20 / 30 | 358.3k / 450.2k / 450.1k / 449.7k | −66.8k / flat | | | topup must stay on |

**Why it works:** the split is what helps, not the room freed at the moment the touch moves.
- A level that holds the whole 200-lot allowance at the old touch (size_top 200 with levels 2, or
  `keep_behind`) blocks re-entry at a freshly improved touch, and loses 71k.
- Shrinking the deep order to top up the touch (`deep_yield`) turns B2 back into a touch-heavy book, and
  loses 133k.
- The gain comes from two things together:
  - half the allowance holds early priority at the next level, so fills at the new touch come from the
    front of the queue;
  - half as much sits at the touch to be run over by large sweeps.

## 2. Spread regimes (spread_episodes.py, full tape)

- Spread ≥ 2 covers 7.0% of the time, in 14.8k episodes.
  - Median episode length is 23 ms (p10 3 ms, p90 148 ms).
  - 16% last under 5 ms and 28% under 10 ms, so they are gone before an order of ours can land.
- 70% close by **continuation**: the opposite side moves in and the mid drifts +0.97 tick by 1–30 s.
  30% close by **refill**: the broken side comes back, with a drift of +0.2.
- The side that closes the gap had the thicker touch 60% of the time. The new inside level is tiny
  (median 14 lots).

**Stepping inside a 2-tick spread loses heavily on round1, on every side choice:**

| variant | Δ vs base |
|---|---|
| `improve_reduce` | −133.7k |
| improve 1 thick | −77.9k |
| improve 2 thin | −131.8k |
| improve 4 continuation | −152.9k |
| improve 5 refill | −36.7k (on top of B2: −12.8k) |
| improve 6 room | −134.4k |

All are 0/10 fold wins with |t| > 11. Part of the cost is giving up our queue priority at the old touch.
The smaller loss on top of B2 fits that, since B2 keeps the old level as its deep order. The rest is
adverse selection while we sit alone inside the spread.

`improve_both` (step inside on both sides only when the spread is 3 ticks or wider, 0.08% of the time)
gives **+3.3k** on the baseline (t 4.1, 9/10, dSd 0.25k) and **+3.3k** on B2 (453,901, t 3.6, 9/10).
The gain is small but consistent.

## 3. Order size, splitting, modify vs cancel

- size_top 200 is right for levels 1:
  - 150: −30.0k, 100: −102.2k;
  - 300 is the same as 200 because the guard clips it to the 200-lot allowance.
- The real size question is how to split the allowance between the touch and one tick behind (section 1).
  The answer is roughly half and half.
- Splitting into several orders at one level (`chunk` 50 / 100) gives **identical PnL**. FIFO treats
  one order of 200 and two of 100 the same, and ladder never partially cancels, so no priority is gained.
- Modify vs cancel to shrink: `deep_yield` modify against cancel differs by 1.6k, which is noise. Shrinking
  only matters when there is a reason to shrink, and every policy tested that shrinks hurts. `modify_order`
  keeps priority (the simulator reduces the order in place), so it is the right tool if a later round needs it.

## 4. max_spread and tight_offset

- `max_spread` 3 / 8 / 20 give identical results (+59 / 0 / 0). Spreads of 3 ticks or more are 0.08% of
  the time, so the rule is harmless.
- max_spread 1, which stops quoting at a 2-tick spread, costs −45.3k on base and −19.5k on B2. Quoting
  through 2-tick spreads is worth money.
- max_spread 2 is within noise (−0.4k base, +0.9k B2). Keep 8.
- `tight_offset` 1 costs **−277.4k** (only 3.7k fills, against 31k). Never use it on round1.

## 5. Cross-tape check

On sample_v2 (single run), q_ladder2 makes 230,883 against 263,512 for the base, which is **−33k**.
On that tape the big uninformed IOCs make the touch itself the edge, so moving half the size behind
hurts. The round1 gain looks specific to round1's microstructure: thick second levels, impact that
grows with order size, and frequent breaks. Keep it behind a parameter and re-test on every new round.

## Recommendations (ranked)

1. **Use levels 2 with an even split of the side's allowance between the touch and one tick behind.**
   Either `size_top 100, size_deep 100` or `top_frac 0.5` (the latter has no constant tuned to the cap).
   - Folds: +25.5k (t 3.6, 9/10). Full tape: +29.2k, at 454.8k. Lower pnl_se, less time at the cap.
   - Per-fold deltas: −0.2, +5.4, +4.6, +2.8, +5.6, +0.1, +0.2, +0.6, +3.8, +2.5k. The gain is uneven
     but almost never negative.
   - Sensitive: an 80/120 split already loses about half the gain, and moving size back to the touch
     loses it all. If impact_mm or another strategy builds on ladder_mm, apply this there and re-A/B,
     because it changes the inventory profile (|pos| at the cap 44% vs 79% of transitions).
2. **Add `improve_both` for spreads of 3 ticks or more.** +3.3k, consistent (t ≈ 4), costs nothing.
   - The B2 + improve_both total is 453.9k on folds.
   - `strategies/marek/q_ladder2.py` has both as defaults.
3. Keep `max_spread` 8, `tight_offset` 0, `topup` 10 (5–30 is flat; 0 is −67k).
4. Do not step inside a 2-tick spread on round1, on any side. Keep the `improve` switches off.
5. Splitting into several orders at one level, and modify vs cancel: no effect. Do not spend more time on them.
6. Before submitting, check the private-tape risk: the split hurts on sample_v2. Round1's private tape
   is presumably the same process as the public one, so ship B2 for round1, then re-validate the split
   at the start of every later round.

Raw outputs: `research/round1/queue/out/sweep{1..4}.txt`, `ab_results.jsonl`, `base.pkl`, `b2.pkl`,
`spread_episodes.csv`.
