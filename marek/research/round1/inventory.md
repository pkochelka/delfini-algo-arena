# Round 1: inventory pinning in ladder_mm

Question: ladder_mm (`strategies/peter/ladder_mm.py`, full round1 PnL 425,639, walk-forward wf01-10 total 425,118,
fold sd 2.6k) sits at |pos| = 100 about half the time (result.json q50 = 99). Does that cost us money, does the PnL
depend on the up-drift (mid 10,000 → 10,030), and can inventory-aware quoting improve robust PnL?

Short answer: the pinning is how the strategy makes money, not a leak. It doesn't hold a directional bet: the
position flips between +100 and -100 every ~0.5 s and averages ~0. Drift explains < 0.3% of PnL. Every
inventory-control variant tried on round1 loses money there. The one useful result is a regime switch that turns
inventory skew on only while our own PnL is falling. It costs nothing on round1 and sample_v2 (it never triggers)
and recovers ~40% of the loss on the adverse `random_stress` tape.

## Scripts (all in research/round1/inventory/)

| file | what |
| --- | --- |
| `mid_cache.py [tape]` | top-of-book series from the tape replay → `data/.cache/<tape>_quotes.npz` |
| `harness.py` | `run_rec` (fold replay that records our fills), `decompose` (PnL split, pinning stats) |
| `analyze_baseline.py` | per-fold decomposition, pin episodes, markout of the flow we miss on the blocked side |
| `sweep_inv.py [variants]` | A/B of `strategies/marek/inv_mm.py` switches on wf01-10 (paired deltas) |
| `other_tapes.py [variants]` | the same variants on full `sample_v2` and `random_stress` |

`strategies/marek/inv_mm.py` is ladder_mm plus switches: `lim_add`, `lim_flip`, `size_add`, `add_back`,
`end_us`/`unwind_s`/`unwind_ioc_s` and `adapt_w_s`/`adapt_thr`/`adapt_skew`/`adapt_lim`. With every switch at its
default it reproduces ladder_mm exactly (425,118 on wf01-10).

## 1. What the pinning looks like (baseline, wf01-10)

PnL decomposition: we take fills against the tape mid. *spread* = Σ q·(mid at fill − fill price). *invPnL* =
Σ pos·Δmid. *liq* = end mark minus mid. *slowN* = mean position over N-second blocks × the block's Δmid, the
directional part of invPnL. These add up exactly to the simulator PnL.

| fold | pnl | spread | invPnL | slow10 | slow60 | slow300 | liq | Δmid | mean pos | mean \|pos\| | time at \|100\| |
| --- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| wf01 | 40,609 | 94,182 | -53,523 | -3,094 | -786 | -117 | -50 | +39.5 | -1.8 | 75.3 | 0.49 |
| wf02 | 36,821 | 93,369 | -56,498 | -3,568 | -562 | 24 | -50 | -4.0 | -0.1 | 78.6 | 0.52 |
| wf03 | 42,495 | 95,134 | -52,598 | -2,436 | -811 | -213 | -40 | +25.0 | 1.8 | 76.3 | 0.51 |
| wf04 | 40,990 | 96,730 | -55,693 | -4,589 | -1,036 | 41 | -48 | +16.0 | -0.8 | 74.4 | 0.49 |
| wf05 | 40,818 | 96,312 | -55,444 | -2,354 | -220 | 25 | -50 | +5.0 | 2.1 | 75.6 | 0.50 |
| wf06 | 45,623 | 99,240 | -53,592 | -3,205 | -1,042 | -217 | -24 | -29.0 | 0.5 | 75.3 | 0.48 |
| wf07 | 44,733 | 98,113 | -53,321 | -2,847 | -958 | -103 | -59 | -1.0 | 2.1 | 76.5 | 0.51 |
| wf08 | 44,711 | 97,157 | -52,396 | -3,061 | -717 | -110 | -50 | -48.0 | 2.5 | 74.6 | 0.49 |
| wf09 | 45,100 | 100,206 | -55,056 | -4,457 | -881 | -333 | -50 | +31.0 | -4.3 | 77.0 | 0.53 |
| wf10 | 43,218 | 98,215 | -54,947 | -2,482 | -179 | 133 | -50 | -5.0 | -1.4 | 75.1 | 0.47 |
| **total** | **425,118** | **968,658** | **-543,068** | -32,094 | -7,193 | **-868** | **-472** | | | 75.9 | 0.50 |

- **Pinned time.** Half the time |pos| = 100 (|pos| ≥ 95: 26-28% long, 24-30% short in every fold, symmetric).
- **Episodes are short.** There were 6,224 long and 6,248 short episodes at |pos| ≥ 95. Median 0.40 s, mean
  0.51 s, p90 1.1 s, max 4 s. The strategy is not stuck. One big taker flips it from +100 to -100: at +100 it
  rests 200 lots on the ask. 2.13M of the 2.65M lots filled reduce or flip the position; 0.52M add to it.
- **Drift dependence is negligible.** The time-averaged position is −4.3..+2.5 lots per fold. The 300 s
  directional term is −868 over the whole tape (0.2% of PnL), and mean pos × Δmid is −288. Fold PnL vs fold Δmid:
  correlation −0.31 on 10 points, which is noise. The 5 up folds made 210.0k and the 5 down folds 215.1k. Being long
  during the up-drift was **not** luck: we were not net long.
- **The real "inventory cost" is adverse selection, not drift.** invPnL is −543k against +969k of spread capture.
  Almost all of it is fast: about −511k sits at horizons under 10 s, −32k between 10 s and 300 s, ~−1k above
  300 s. That is the queue-position and impact topic (see impact_mm docstring), not inventory management.
- **End liquidation costs 47 per fold** (100 lots × half a tick), −472 over 10 folds and −50 on the full tape.
  That is 0.1%, well under the noise.
- **The blocked side is not a missed opportunity.** Below is the maker markout (mid(t+h) − touch, our direction,
  weight min(qty,100)) of every tape aggressive order, split by our position when it arrived:

| flow arriving on… | orders | lots | 0 | 1 s | 5 s | 30 s | 120 s |
| --- | --: | --: | --: | --: | --: | --: | --: |
| blocked side (we're pinned, can't quote) | 25,226 | 1.51M | 0.433 | **-0.031** | -0.008 | -0.019 | -0.054 |
| adding side, not pinned | 22,665 | 1.36M | 0.440 | 0.065 | 0.059 | 0.083 | 0.137 |
| reducing side | 46,603 | 2.78M | 0.458 | **0.310** | 0.302 | 0.301 | 0.240 |

  The flow we miss while pinned is worth ~0 after one second. It is continuation flow: the same direction that just
  filled us 100 lots, so it is the toxic part. The ±100 cap blocks it for free. The reducing-side flow is the
  money: +0.30 ticks/lot, which is why the 200-lot flip order works.

## 2. Inventory-aware variants (wf01-10, paired vs baseline)

`d_sd` is the sd of the per-fold paired delta. It is much smaller than the 2.6k fold sd because every variant
replays the same tape, so a delta beyond ~2·d_sd·√10 is real.

| variant | total | Δ | d_sd | spread | invPnL | slow300 | up 5 | down 5 | mean \|pos\| | lots |
| --- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| base (= ladder_mm) | 425,118 | 0 | 0 | 968,658 | -543,068 | -868 | 210,012 | 215,106 | 75.9 | 2.65M |
| add_back 1 tick @ \|pos\|≥80 | 423,402 | -1,716 | 302 | 960,270 | -536,392 | -877 | 209,477 | 213,925 | 75.2 | 2.62M |
| unwind last 5 s (passive) | 423,484 | -1,634 | 175 | 964,892 | -541,408 | -878 | 209,001 | 214,483 | 75.7 | 2.64M |
| size_add 50 | 419,950 | -5,168 | 218 | 954,112 | -533,690 | -857 | 207,061 | 212,889 | 75.6 | 2.61M |
| unwind last 30 s (+IOC 2 s: same) | 414,209 | -10,909 | 422 | 943,794 | -529,586 | -944 | 204,807 | 209,402 | 74.1 | 2.58M |
| lim_add 75 | 412,653 | -12,465 | 518 | 926,104 | -512,983 | -779 | 203,364 | 209,289 | 74.6 | 2.53M |
| skew_lots 50 (= add_back 1 @50) | 411,361 | -13,757 | 738 | 918,354 | -506,536 | -744 | 203,249 | 208,112 | 72.7 | 2.48M |
| size_add 25 | 410,859 | -14,259 | 578 | 930,626 | -519,299 | -784 | 202,311 | 208,548 | 75.4 | 2.55M |
| lim_add 50 | 404,806 | -20,312 | 662 | 900,042 | -494,780 | -712 | 199,586 | 205,220 | 74.2 | 2.46M |
| lim_add 25 | 399,304 | -25,814 | 914 | 886,164 | -486,410 | -741 | 197,359 | 201,945 | 74.3 | 2.42M |
| lim_flip 75 | 390,686 | -34,432 | 698 | 890,068 | -498,969 | -826 | 192,769 | 197,917 | 66.2 | 2.47M |
| unwind last 120 s | 381,431 | -43,687 | 982 | 873,118 | -491,686 | -1,035 | 189,273 | 192,158 | 68.3 | 2.38M |
| skew_lots 25 | 379,075 | -46,043 | 1,304 | 841,640 | -462,129 | -746 | 187,528 | 191,547 | 68.7 | 2.27M |
| lim_flip 50 | 336,614 | -88,504 | 966 | 786,239 | -449,261 | -899 | 167,159 | 169,455 | 56.7 | 2.22M |
| improve_reduce | 291,457 | -133,661 | 2,008 | 844,338 | -552,509 | -901 | 145,467 | 145,990 | 72.2 | 3.11M |
| flat_at 100 → 50 (IOC) | -59,786 | -484,904 | 2,663 | | | | | | 47.8 | (165k breach penalty) |
| flat_at 100 → 0 (IOC) | -547,125 | -972,243 | 5,297 | | | | | | 27.4 | (239k breach penalty) |
| adapt 60/120/300 s, skew 25/50, lim 50 | 425,060 – 425,118 | 0 … -58 | ≤12 | | | | | | 75.9 | 2.65M |

Switches:
- `lim_add`: max |pos| the adding side may reach.
- `lim_flip`: how far past flat the reducing side may take us.
- `size_add`: lots on the adding side while holding a position.
- `add_back`: the adding side quotes N ticks behind the touch once |pos| ≥ X.
- `unwind`: from end − N s, cancel everything and rest only the reducing side up to flat.
- `adapt`: see section 3.

What the sweep shows:
- **Every way of shrinking the adding side or the flip loses money**, and the loss grows with how hard it bites.
  It is far outside the noise: lim_add 50 is −20k with a paired d_sd of 0.7k, and all 10 folds are negative. Less
  adding volume means less spread capture, and the adverse selection we avoid is smaller than the spread we give up.
  This matches the markout table: adding-side flow is still +0.06..+0.14 ticks/lot.
- **Limiting the flip (`lim_flip`) is the worst.** The flip fills are the reducing-side +0.30 ticks/lot flow.
- **skew_lots / improve_reduce / flat_at hurt on round1 too**, matching the sample_v2 result. flat_at also breaches
  the limit: its IOC is outside the cap guard while the resting orders stay. Keep it off, or fix the guard before
  any reuse.
- **End-of-tape unwind loses.** Passive unwinding in the last 5 s gives up ~160/fold of trading and queue priority
  to save ~47/fold of liquidation. The private tape's length is unknown anyway. Accept the ~0.5 tick × 100 lots
  markout at the end.
- **No variant changes the up/down split.** Every variant loses about equally on rising and falling folds, as
  expected when the position carries no drift exposure.

## 3. Robustness to other data (full tapes, one run each)

| tape | variant | pnl | spread | invPnL | slow300 | Δmid | mean pos | mean \|pos\| |
| --- | --- | --: | --: | --: | --: | --: | --: | --: |
| sample_v2 | base | 263,512 | 199,458 | +64,104 | 1,040 | +4.0 | 0.1 | 76.9 |
| sample_v2 | lim_add 50 | 248,479 | 189,222 | 59,274 | 898 | | -0.5 | 74.1 |
| sample_v2 | size_add 50 | 262,175 | 198,986 | 63,208 | 981 | | 0.0 | 76.8 |
| sample_v2 | add_back 1 @80 | 261,841 | 197,454 | 64,437 | 994 | | 0.0 | 75.5 |
| sample_v2 | skew_lots 50 | 249,560 | 185,376 | 64,202 | 924 | | 0.2 | 70.9 |
| sample_v2 | lim_flip 75 | 251,898 | 190,576 | 61,341 | 1,037 | | 0.0 | 68.6 |
| sample_v2 | all adapt variants | 263,512 | = base (never triggers) | | | | | |
| random_stress | base | **-18,561** | 9,734 | -28,232 | -4,077 | +34.5 | **-13.7** | 77.8 |
| random_stress | lim_add 50 | -15,050 | 8,024 | -22,867 | -3,635 | | -9.8 | 64.0 |
| random_stress | size_add 50 | -19,411 | 8,920 | -28,244 | -3,781 | | -15.8 | 78.1 |
| random_stress | add_back 1 @80 | -16,109 | 9,976 | -26,072 | -3,829 | | -13.1 | 73.6 |
| random_stress | skew_lots 50 | **-7,115** | 9,194 | -16,189 | -3,351 | | -12.8 | 58.5 |
| random_stress | lim_flip 75 | -18,587 | 8,854 | -27,288 | -3,955 | | -15.5 | 74.7 |
| random_stress | adapt 60 s skew50 | -11,920 | 9,632 | -21,514 | -3,348 | | -14.4 | 65.1 |
| random_stress | adapt 120 s skew50 | -12,336 | 10,019 | -22,283 | -3,191 | | -11.2 | 65.1 |
| random_stress | adapt 300 s skew50 | **-10,888** | 8,812 | -19,613 | -3,707 | | -15.2 | 63.5 |
| random_stress | adapt 120 s skew25 | -12,076 | 8,795 | -20,850 | -3,466 | | -15.1 | 56.9 |
| random_stress | adapt 120 s lim50 | -16,340 | 8,703 | -24,977 | -3,475 | | -12.8 | 68.5 |
| random_stress | adapt 120 s skew50+lim50 | -11,755 | 8,680 | -20,135 | -3,059 | | -9.9 | 59.6 |

- On sample_v2 the inventory term is even positive (+64k): the flow there mean-reverts. Inventory control only
  costs money there.
- `random_stress` is where the ladder breaks. Spread capture is small, adverse selection is large, and the position
  leans against the trend (mean pos −13.7 while mid rises 34.5). The slow, drift-driven part is −4.1k, about 22% of
  the loss. The rest is fast adverse selection. Here inventory skew helps: always-on skew_lots 50 recovers +11.4k.
- **The `adapt` regime switch is a free hedge.** It tracks our own mark-to-mid PnL every second. When the change
  over the last W s is < 0, it applies `skew_lots = adapt_skew` (and optionally `lim_add = adapt_lim`).
  - On round1, PnL grows ~35/s, so a 120-300 s window never goes negative: Δ = 0 exactly. At 60 s it triggers
    briefly: −58, d_sd 12.
  - On sample_v2 it never triggers.
  - On random_stress it recovers +6.2k to +7.7k (300 s window best), about 60% of what always-on skew gets there.
  - Caveat: random_stress is one 2,000 s run. Its inventory noise is about 100 × 0.92 × √2006 ≈ 4k, so the
    +7k is ~1.7σ. The direction agrees with always-on skew, but the size is not well measured.

## Recommendations (ranked by expected robust gain)

1. **Keep the ±100 pinning and the 200-lot flip order as they are (cap 100, size_top 200, no inventory throttle).**
   It is the edge: +0.30 ticks/lot on reducing flow. The blocked side costs nothing (≈0 markout), and the PnL has
   no meaningful drift exposure (mean position ~0, slow part 0.2% of PnL). Every throttle tried costs 1.6k-134k on
   round1, 4-34× the paired delta sd. This also answers the robustness worry: the round1 PnL does **not** depend on
   being long into the up-drift.
2. **Add the `adapt` regime switch as cheap insurance for the private tape and later rounds:** `adapt_w_s = 300`,
   `adapt_thr = 0`, `adapt_skew = 50` (inv_mm.py; ~15 lines to port into ladder_mm / impact_mm).
   - Expected gain on round1-like data: 0 (it never triggers).
   - On a tape where the ladder is losing (random_stress): about +40% of the loss recovered.
   - Measured risk: none on round1 or sample_v2. It could hurt on a tape whose 300 s PnL dips below 0 while the
     strategy is still profitable overall, so a larger window or a negative `adapt_thr` is the conservative choice.
   - Weakness: the evidence for the gain comes from a single stress tape.
3. **Don't unwind at the end.** Liquidation costs ~50 per run (100 lots × half a tick). Passive unwinding in the
   last 5-120 s costs 1.6k-44k over the folds.
4. **Leave skew_lots, improve_reduce, flat_at, lim_add, lim_flip and size_add off for round1-like data.**
   flat_at also needs a guard fix (IOC outside the cap guard → breaches) before it can be reused anywhere.
5. Where the remaining money is: the inventory term (−543k) is almost entirely sub-10-second adverse selection.
   That makes it a fill-selection problem (which takers fill us, queue position, impact), which belongs to the
   impact/queue topics, not to inventory management.
