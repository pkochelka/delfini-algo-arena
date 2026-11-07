# queue_mm hypotheses

Hypotheses for `queue_mm.py` (a copy of `ladder_mm.py` with new flags), with what was tested and the result.
Add new ones under **Open**. Every flag stays in PARAMS (off by default unless marked adopted).

**Data:** round1 only. Folds in `data/splits_r1/wf`. Robustness set: wf03, wf07 and wf11, each under one
`augment.py` attribute change at a time, in `data/splits_r1/aug`:
`orig pscale2 tscale0.25 tscale4 vscale0.3 vscale3 drift30 noise1 takers3 shocks3`.
`mirror` gives exactly the orig PnL (the strategy is symmetric), so it was dropped.

**Score:** mean PnL over the 3 folds per tag. The tables report MEAN over tags and WORST tag.
Each fold is a whole file traded from t=0 (no trade_start).

**Tools:** see [Tools](#tools) at the end.

**Baseline (v1):** ladder_mm tuned (`size_top` 99, `size_deep` 50, `levels` 2, `topup` 6, `skew_lots` 100).
Robustness MEAN +30.6k, WORST +16.1k (vscale0.3). The 12 folds average +27.7k.

## Facts established (round1, wf09, baseline params)

**Markout averages over fills** (ticks/lot): +0.52 at the fill, +0.34 after 0.1 s, +0.18 after 1 s.
Adverse selection eats about 2/3 of the half-spread.

**By market qty left on our side after the fill:**

| Share left | Share of lots | Markout 1 s |
| --- | --- | --- |
| < 5% | 12% | −0.24 |
| > 80% | 20% | +0.50 |

**By queue position before the fill** (market qty ahead of our order):

| Ahead of us | Share of lots | Markout 1 s |
| --- | --- | --- |
| 0 (we're first) | 37% | +0.36 |
| 1–100 | 49% | +0.23 |
| 100–300 | 10% | +0.17 |
| 300–1500 | small | +0.03 … −0.04 |

**By order age at the fill:**

| Age | Markout 1 s |
| --- | --- |
| < 0.1 s | +0.11 |
| 0.1–0.5 s | +0.16 |
| 0.5–2 s | +0.35 |
| 2–10 s | +0.32 |

**Takeaways:**
- The edge comes from **queue seniority**: old orders that market cancels have pushed toward the front.
- PnL is **capacity-bound**: it scales almost linearly with `cap` (60 gives 21.0k, 100 gives 30.6k).
  A cap of 200 breaches, about −3.8M in penalties.

## Tested

| # | Hypothesis | Flag | Result | Verdict |
| --- | --- | --- | --- | --- |
| 1 | Pull a side when its share of the touch qty is small (thin side = toxic fills) | `imb_pull` 0.1 / 0.2 / 0.3 | 12 folds: 27.7k becomes 20.5k / 13.8k / 8.7k | Rejected: cancels destroy queue seniority |
| 2 | Rest deeper (more levels, start behind the touch) to be early in line when the price arrives | `levels` 3–4, `tight_offset` 1, `size_top` 0 | All worse: 8.5k–23.8k vs 27.7k | Rejected: the touch is the place |
| 3 | Cancel only the touch order when its level is about to vanish (market qty < X, other side ≥ ratio·X) | `vanish_qty` 30 / 50 / 150 | MEAN 30.5k / 30.2k / 28.0k vs 30.9k; thin book +0.8k | Rejected: small loss, the level often refills |
| 4 | Step inside a 2-tick spread to be first in a fresh level (thick / thin / reducing side) | `improve` 1 / 2 / 3, `improve_reduce` | MEAN 28.4k / 20.0k / 20.6k / 20.6k; pscale2 down to −34k … −38k | Rejected: being first in a fresh level is the toxic case |
| 5 | Grid search of sizes and topup across tags | `size_top`, `size_deep`, `levels`, `topup` | One level × 200: pscale2 +57k, vscale3 +45k, vscale0.3 +4.9k. Two levels 99/50: thin +16k, pscale2 +45k. `topup` 20 slightly beats 6 (MEAN 30.8k) | No single setting wins every tag |
| 6 | "One level behind" should follow the market's price grid, not 1 tick (on a 2-tick grid a deep order becomes a fresh, toxic level) | `grid_unit` 0 = learn the most common gap between occupied levels | pscale2 45.0k becomes **59.5k**, other tags unchanged. MEAN 32.3k, WORST 16.3k | **Adopted** |
| 7 | Size the touch order from the EWMA market taker size (thick books want more) | `size_k` 0.6 / 0.8 / 1.0 | MEAN 31.7k / 31.2k / 30.2k; vscale3 37.2k becomes 31.8k | Rejected |
| 8 | Drop the deep level when our cap is a small slice of the touch (EWMA touch > 25·cap) | `deep_ratio` 25 | vscale3 37.2k becomes 32.3k (worse); others unchanged | Rejected as is: the thick book wants one level **and** size 200 (`size_top` 200 alone gives vscale3 44.9k but hurts everywhere else) |

| 8b | Regime switch on touch/cap. Thick book (EWMA touch > 25·cap): one level × 200 lots. Otherwise 99/50 on 2 levels | `deep_ratio` 25, `thick_size` 200 | vscale3 **45.3k** (from 37.2k), other tags unchanged. On orig/pscale2/vscale0.3/vscale3/takers3: MEAN 36.6k vs 32.2k without the switch, WORST 16.3k. `deep_ratio` 15 is slightly worse (orig touch ≈ 13·cap, too close) | **Adopted** |

| 9 | Thick-book threshold: where between 1× and 3× depth does one level start winning? (orig touch ≈ 13·cap, 1.7× ≈ 22·cap, 2× ≈ 26·cap) | `deep_ratio` 15 / 18 / 25 / 35; new augments vscale0.6/1.7/2 | vscale1.7: 18 gives **37.6k** vs 33.6k at 25. vscale2: 40.2k vs 39.0k. 12 real folds: 18 is identical to 25 (never switches); 15 switches on round1 (mean 27.5k vs 27.6k) | **Adopted 18** |
| 10 | Thin book (worst tag): more levels, other sizes or topup when the touch is small | `levels` 3, `size_top`/`size_deep` 60/60, 50/100, 40/40×3, `topup` 5/40 on vscale0.3/0.6, pscale2_vscale0.3, vscale0.3_takers3 | All worse than the defaults (MEAN 22.9k vs 18.1–22.7k) | Rejected. The thin book's PnL is low because 0.3× volume means 0.3× flow: per lot of market aggressive volume it earns about 0.060 vs 0.033 on orig |
| 11 | Don't *send* new touch orders on the thin side (cancel nothing, so no seniority is lost) | `gate_share` 0.1 / 0.2 / 0.3, `gate_all` | MEAN 45.5k becomes 36.9k / 34.2k / 31.1k / 34.0k on orig/pscale2/vscale0.3/vscale3/takers3/pscale2_vscale3 | Rejected: a new order on a thin side is near the front of a short queue, so those fills are good |
| 12 | Split the touch order into chunks so a partial fill keeps the senior remainder | (none) | Not run: price-time priority already keeps a partially filled order's remainder at the front, so splitting only adds messages | Dropped (no mechanism) |

| 13 | Thick-book switch warm-up: the touch EWMA can fire early when trading from t=0 | `deep_min_n` 5000 / 20000, `touch_n` 5000 | 5000: orig back to 29,795 (from 29,716), vscale1.7 37.6k and vscale3 45.2k kept. 20000 loses ~0.5k at 1.7×/3× | **Adopted 5000** |
| 14 | New shift: bigger / smaller sweeps with the same book (augment `iocscale` 0.5/2/3) | defaults vs one level × 200 | iocscale2: defaults 51.8k, one level **57.2k**. iocscale3: 69.0k vs **81.2k**. iocscale0.5: two levels 16.3k vs 10.5k. The book-depth switch doesn't see this (touch unchanged) | Finding: the regime variable is **sweep size relative to cap**, see #16 |
| 15 | New shift: queue turnover (augment `cscale`: cancels at 0.5× / 2× the original order life) | defaults vs baseline vs one level | cscale0.5 (faster turnover): 58.8k. cscale2 (slower): 13.9k (baseline 11.9k, one level 13.9k) | Finding: seniority comes from the market cancelling ahead of us; turnover is the biggest single driver of PnL. Nothing to tune; the depth switch already helps at cscale2 (slow cancels pile up the touch) |
| 16 | One-level switch on EWMA market taker size > ratio·cap (OR with the depth switch). Measured: orig 1.55·cap, vscale1.7 2.64, iocscale2 2.73, vscale3 4.66 | `sweep_ratio` 1.9 / 2.1 / 2.4; 2.1 with `deep_ratio` 0 | 1.9–2.4 all identical: iocscale2 **57.4k**, iocscale3 **81.1k**, rest unchanged; MEAN over 12 tags 45.5k vs 44.1k. Without the depth switch cscale2 drops to 12.0k | **Adopted 2.1** (flat plateau; keep both switches) |
| 17 | Keep senior orders when the price moves away (don't cancel up to K grid steps behind the ladder) | `keep_levels` 1 / 3 | MEAN over 8 tags 43.1k becomes 35.7k / 19.3k. With 1 it's identical in two-level regimes, but in one-level regimes vscale3 drops 45.2k to 32.0k | Rejected: stale orders hold capacity the touch needs (capacity-bound, see Facts) |
| 18 | Top up only into short queues: a top-up joins the back, where fills mark out ~0 | `topup_ahead_max` 300 / 800 | MEAN 38.8k / 41.0k vs 43.1k | Rejected: back-of-queue fills are low-edge but positive, and the capacity is otherwise idle |
| 19 | Use idle capacity on the reducing side: the touch order grows to all free room minus the deep order's share | `fill_room` 1 | MEAN 41.7k vs 43.1k (orig 27.6k vs 29.8k) | Rejected |
| 20 | Inventory skew re-check on the robustness set | `skew_lots` 0 / 50 / 100 / 200 | 0 = 100 = 200 exactly; 50 is −1.3k | **Default set to 0.** Under the cap guard \|pos\| ≤ 100, so int(pos/100) is nonzero only at ±100, where the adding side has no room anyway: 100 was a no-op |
| 21 | Wide-spread freeze (`max_spread`, in grid steps): does the cutoff matter? | 3 / 8 / 30 | Within ±30 of each other | No effect (kept 8) |
| 22 | `topup` re-check under the final switches (25 tags, 3 folds) | 6 / 10 / 20 | MEAN 37,412 / **37,426** / 37,382. 12 folds: 27,724 / 27,716 / 27,630 | **Adopted 10** (best on robustness; ties baseline on round1) |

**Combined shifts** (new augments, defaults vs the baseline params):

| Tag | Baseline | queue_mm |
| --- | --- | --- |
| pscale2_vscale0.3 | 17.5k | **32.4k** |
| pscale2_vscale3 | 62.4k | **90.4k** |
| vscale0.3_takers3 | 17.6k | 17.8k |
| tscale4_vscale3_drift30 | 36.8k | **44.5k** |

The adaptations add up when shifts combine; nothing broke.

**Pattern across #1, #3, #4, #11:** every attempt to be *selective* about fills (pull, gate, step inside the spread) lost.
In round1-like markets, fill quantity at senior queue positions beats fill selectivity. A new filter idea
should only remove fills, never reset or block queue positions, and should be checked against that.

## Current defaults

Adopted: #6 `grid_unit` 0, #8b/#9 `deep_ratio` 18 + `thick_size` 200, #13 `deep_min_n` 5000, #16 `sweep_ratio` 2.1, #20 `skew_lots` 0, #22 `topup` 10.

**Round1 itself** (neither switch fires here, so expect a tie):

| Run | Baseline (ladder tuned) | queue_mm |
| --- | --- | --- |
| 12 walk-forward folds | mean +27,724 (sd 1,083) | mean +27,716 (sd 1,079) |
| Held-out test 9000–12000 s | +117,290 | **+117,551** |

**Full robustness scoreboard** (25 tags × wf03/07/11, mean PnL per tag): baseline = ladder tuned params, `topup` 20 run.

| Tag | Baseline | queue_mm | | Tag | Baseline | queue_mm |
| --- | --- | --- | --- | --- | --- | --- |
| orig | 29,941 | 29,795 | | takers3 | 32,289 | 32,159 |
| pscale2 | 45,182 | **59,525** | | makers3 | 29,639 | 29,480 |
| tscale0.25 | 26,482 | 26,355 | | flicker3 | 29,917 | 29,762 |
| tscale4 | 31,090 | 30,781 | | shocks3 | 30,921 | 30,815 |
| vscale0.3 | 16,057 | 16,290 | | iocscale0.5 | 16,092 | 16,281 |
| vscale0.6 | 25,165 | 25,147 | | iocscale2 | 51,920 | **57,410** |
| vscale1.7 | 34,035 | **37,636** | | iocscale3 | 69,152 | **81,085** |
| vscale2 | 35,094 | **40,203** | | cscale0.5 | 58,885 | 58,756 |
| vscale3 | 37,198 | **45,191** | | cscale2 | 11,890 | **13,858** |
| drift30 | 29,402 | 29,412 | | pscale2_vscale0.3 | 17,500 | **32,439** |
| drift-30 | 29,291 | 29,206 | | pscale2_vscale3 | 62,438 | **90,262** |
| noise1 | 30,578 | 30,428 | | vscale0.3_takers3 | 17,644 | 17,840 |
| | | | | tscale4_vscale3_drift30 | 36,756 | **44,429** |
| **MEAN** | **33,382** | **37,382** | | **WORST** | 11,890 | 13,858 |

With the adopted `topup` 10 the MEAN is 37,426 and the WORST 13,936.
The 0.3–0.5% dips on unswitched tags came from `topup` 20 (#22).

## Open

- [x] Thick-book switch warm-up: done as #13.
- [x] Bigger sweeps and queue turnover: done as #14, #15 and #16.
- [ ] Wider spreads on the same tick grid: no augment yet (needs a book-aware step that thins the touch).
- [ ] The switch thresholds (`deep_ratio` 18, `sweep_ratio` 2.1) were placed from augmented round1 only. A real round2 tape is the first true out-of-sample check: log `aggr`/`touch_q` there before trusting them.
- [ ] Queue turnover (#15) is the biggest PnL driver and nothing adapts to it yet. Is there a lever (e.g. size, or levels, vs the measured market cancel rate)?

## Tools

All in `strategies/peter/`, run from the repo root:

| Script | What it does |
| --- | --- |
| `robust_eval.py strategy.py '{...}' ... [--tags a,b] [--folds wf03,wf07,wf11]` | The robustness table above (tags = augment file suffixes, `orig` = the plain fold) |
| `fold_eval.py strategy.py '{...}' ...` | The 12 walk-forward folds with trade_start |
| `test_eval.py [--strategy path] '{...}' ...` | Held-out test 9000–12000 s (default strategy queue_mm) |
| `fill_probe.py strategy.py data.csv out.npz ['{...}']` + `fill_markout.py out.npz` | Per-fill markouts by queue position / age / share |
| `augment.py ... --only <tags>` | New presets: vscale0.6/1.7/2, the combined tags, iocscale0.5/2/3 (aggressive volumes only), cscale0.5/2 (cancel delays) |
