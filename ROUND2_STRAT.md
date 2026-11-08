# ROUND2_STRAT: what the "Manhattan" volatility chart means and what to build

Round 2 tape: `data/round2.csv`, 6000 s. Arrival-process research: `marek/research/round2/flow/README.md`. Read that first; this file turns it into a strategy plan.

Written 2026-11-07. Numbers marked *measured* come from the tape or our baseline logs. Numbers marked *estimate* are back-of-envelope figures to be confirmed by a run.

## 1. What the chart shows

The rolling σ over 100 s has two kinds of shape. They come from two different mechanisms:

| Shape | Example times (s) | Cause |
| --- | --- | --- |
| **Flat-topped boxes, exactly 100 s wide** | 1560–1660, 2264–2364, 2467–2567, 3423–3523, 4825–4925, 5398–5498 | A single **shock jump** in S enters the 100 s window and leaves it 100 s later. Box height ≈ \|J\| / 10 on top of the base. Example: the +114.5 jump at 5398 s gives a box at about 11.5. |
| **Rounded humps** | 2017–2159, 3625–3737, 4264–4380 | The **volatile regime**: taker rate ×1.5, impact c ×5, spread about 1.9. Dwell about 44 s on average. |
| Floor at about 1 | most of the tape | The calm regime, which is round 1 with c ≈ 0.030 |

The boxes are an artefact of the window. Each one is a single price jump, not a period of high volatility, and a 10 s or 1000 s window would make them 10 s or 1000 s wide. *Measured*: every box starts at a shock onset in `flow/shock_events.csv` plus 3 s.

The chart therefore shows two things we can trade on:

1. **Shocks: 27 scheduled price jumps** with a fixed 3.0 s warning. Maker inserts drop to 30% for exactly 3 s, then S jumps by N(0, ≈43) ticks.
2. **A two-state regime, calm or volatile**, which the book reveals within about 1 s (per-second spread > 1.5 classifies 96.7% of seconds correctly).

## 2. Where our round 1 strategies lose money on round 2 (*measured*)

Full-tape PnL from `marek/research/round2/baseline/*.log`, split by period. Shock = onset to onset + 20 s; volatile and calm come from the HMM segmentation in `flow/taker_regimes.csv`. Script: `marek/research/round2/baseline/decomp.py`, built on the 1 s `pnl.csv` series.

| Strategy | Calm (4667 s) | Volatile (793 s) | Shock windows (540 s) | Total |
| --- | ---: | ---: | ---: | ---: |
| `s_ladder` | **+150.0k** | −253.4k | −91.1k | −194.5k |
| `sig_ladder` | +146.2k | −314.0k | −65.4k | −233.2k |
| `robust_ladder` | +53.3k | −18.5k | −10.1k | +24.6k |

`s_ladder` within the shock and volatile windows:

| Shock phase (relative to onset) | PnL | | Volatile phase (since segment start) | PnL |
| --- | ---: | --- | --- | ---: |
| −1 to +3 s (drought) | +4.5k | | first 2 s | −9.6k |
| **+3 to +5 s (the jump)** | **−76.5k** | | 2–10 s | −41.3k |
| +5 to +10 s (burst) | −10.3k | | **after 10 s** | **−240.7k** |
| +10 to +20 s | −9.8k | | | |

What the split says:

- **Calm already works.** `s_ladder` earns +150k in calm, about 32 per second, roughly its round 1 rate. Don't touch the calm logic first.
- **The shock loss is concentrated in the 2 s after the jump**, and the 3 s warning comes before it. Resting stale quotes get run over by the jump.
- **The volatile loss is steady-state**, not a detection-lag problem: 95% of it comes after the first 10 s of a segment. A detector with 1–2 s lag captures almost all of it.
- **Upper bound for "calm-only" trading:** +150k calm + 0 elsewhere ≈ +150k, six times `robust_ladder`. Everything in section 3 is about getting from −194k to at least that, then above it.

## 3. Strategies to try, in priority order

### A. Shock guard: pull before the jump (*estimate*: +75k to +90k)

1. **Detect the drought.** A 0.5 s window with fewer than 50 maker inserts (normally 89 ± 9.4) found all 27 shocks with no false positives. Count `market` maker inserts from the book diff, or count new price levels appearing. A 0.25 s window should catch it in about 0.3 s; check its false-positive rate.
2. **Cancel every resting order by onset + 2.5 s** and send nothing until the jump has happened (first maker inserts at a new level, or onset + 3.1 s at the latest).
3. **Flatten inventory during the drought** if it is cheap. Held inventory is zero-mean risk: jumps are symmetric, so it adds variance, not expected loss. Its sd is 100 × 43 ≈ 4.3k per shock, about 22k per tape. The expected loss comes from our resting quotes on the wrong side. Test two variants: (a) quotes pulled only, and (b) quotes pulled plus a reduce-only IOC or an aggressive passive flatten in the drought. Choose on the z-score trade-off: variance matters because a negative round scores 0.
4. **Use the timing prior:** gaps run 147–296 s, roughly U(150, 300). Within 140 s of the last onset a shock is almost impossible, so the detector can be relaxed. After that the hazard rises, so tighten the detector or lower the cap slightly. This is secondary, since detection is already perfect on this tape.

### B. Volatile regime: step aside first, then try to trade it (*estimate*: +230k to +250k from stepping aside)

1. **Detect** the volatile regime from the spread (> 1.5 averaged over 1 s) or the width of the maker envelope (> 2.5). Exit with hysteresis, for example 3 s with spread < 1.2, since exiting late costs calm PnL.
2. **Baseline action: no new quotes, keep only flattening orders.** `robust_ladder` loses only 18.5k there, so going flat is close to free.
3. **Then try to make money in it.** The volatile regime is round 1 scaled up:
   - Impact c ≈ 0.13 (×5), takers cross about 2.5 ticks through S, and the spread is about 1.9.
   - Run `s_ladder` with a regime-switched `s_c` (0.13) and thresholds scaled by the regime's volatility. The edge per fill is ≈ dist − c·√(volume before us). A median taker (q ≈ 60) moves S by about 1 tick, so quote at dist ≥ 2–2.5 ticks from S.
   - **Sniping should pay more here.** After each taker, S jumps about 1 tick, and the slow makers' quotes go stale for 20–40 ms, against our 5 ms. `s_take` with the correct c is the cleanest thing to test.
   - Use a smaller cap in volatile, for example 30–50, so one bad sequence can't cost much.
4. Grid: {flat, wide quotes, snipe only, wide quotes + snipe} × cap {25, 50, 100}. Keep whichever beats "flat" on per-segment paired deltas.

### C. Snipe the stale book right after the jump (*estimate*: +30k to +60k, high variance)

After the drought, the old book is still resting (about 1/8 depth, touch about 113 lots), and new makers quote immediately at the new level. For |J| ≥ 3 ticks every stale level between old and new S is free money, and we are faster than the stale makers and the takers in the burst.

- **Trigger:** the first maker inserts after onset + 3.0 s whose prices sit ≥ 3 ticks from the old mid on one side. That gives the direction and the new S level.
- **Action:** IOC through the stale side up to the cap room, limit ≈ new S − 1. Priority A leaves us flat, so the room is the full 100.
- **Unwind:** quote the reducing side at the new S during the burst. Takers come at about 40/s on both sides, so it fills fast, and PnL is marked to mid anyway.
- *Estimate*: 100 lots × about 15–25 ticks of average edge × the ~20 shocks with \|J\| ≥ 3 ≈ 30k–50k. The risk is misreading the direction from a few noisy inserts, so require 2 or more inserts that agree.
- The jump direction is **not** predictable before the jump (all drought features have |corr| < 0.3, n = 27). Don't pre-position.

### D. Fix the S estimator for round 2 (prerequisite for B and C, and maybe for calm)

- The `s_ladder` estimator `S += 0.026·√q` has no jump term and one c. Fitting the round 1 model to round 2 gives `mid_minus_S_sd = 139` ticks (`r1_model_fit.json`), so it is badly off.
- Needed:
  1. Re-anchor S at each shock jump, using the new makers' centre.
  2. Regime-switched c (0.030 calm, 0.13 volatile).
  3. Keep the touch-bracketing correction already in `s_ladder`.
- Check whether calm has extra diffusion beyond c·√q: calm R² is 0.80 against 0.95 in round 1. If it does, a slow pull towards the maker-insert centre is needed. The `s_dynamics/` work (`full_smooth.py`) should answer this.
- Then re-run the oracle-ceiling check (`APPROACH.md` §2) on round 2.

### E. Smaller ideas, after A–D

- **Liquidity in the burst:** after the jump, about 80 extra symmetric takers arrive within 3 s, the book is thin and the spread is wide. Once S is re-anchored, quoting at the new S could turn the −10k burst window positive.
- **Calm retune:** round 2 calm has c 0.030 rather than 0.026 and a slightly wider spread (1.07). Retune `s_pull` / `s_take` on calm-only folds.
- **Regime-aware kill switch:** keep `robust_ladder`'s equity kill switch as a last line of defence, but its thresholds should key off the detected regime rather than trailing PnL.

### Not worth trying (evidence in `flow/README.md`)

- Trading the chart itself (momentum or mean reversion on rolling σ). The boxes are artefacts of single jumps.
- Directional bets: taker side is a coin flip in both regimes, with no meta-orders and no momentum.
- Predicting regime switches. Durations look geometric, calm → volatile is p ≈ 0.004/s, and a shock does not significantly raise the switch probability (3/27 against 1.6 expected).

## 4. Build plan

One new file, `strategies/marek/r2_ladder.py`, a subclass or copy of `s_ladder` with every mechanic behind a PARAMS flag. With all flags off it must match `s_ladder` exactly (−194,451 on the full tape).

| Step | Flag(s) | Pass criterion |
| --- | --- | --- |
| 1 | `shock_guard` (detect, cancel, wait) | Jump window −76k → about 0; no false triggers on the tape |
| 2 | `vol_mode = flat` (detector + no new quotes) | Volatile −253k → about −20k or better; calm unchanged |
| 3 | `shock_flatten` | Lower PnL variance across shocks; mean not worse |
| 4 | S estimator: jump re-anchor + regime c | S error sd vs the `s_dynamics` smoothed S; calm PnL not worse |
| 5 | `vol_mode = quote / snipe` | Beats `flat` on per-segment paired deltas |
| 6 | `jump_snipe` | Positive mean per shock, and no shock worse than −2k |
| 7 | burst quoting, calm retune | Paired fold deltas |

## 5. Validation (the tape is small in events)

- **Only 27 shocks and 17 volatile segments.** Anything tuned per shock overfits immediately.
- **Build a round 2 generator** first: `r1_synth.py` plus the HMM regime (c, taker rate, offsets, maker width switched) plus scheduled shocks (U(150, 300) gaps, 3 s drought at 0.30× maker rate, N(0, 43) jump, the burst). Every parameter is in `flow/README.md`. Generate 10 or more seeds and treat them as the main test set.
- **Walk-forward folds on round 2** (`strategies/peter/split_data.py`). Compare variants on paired per-fold deltas, as in round 1.
- **Stress:** run on round1 (which has no shocks or regimes, so the new flags must cost about nothing there), sample_v2 and random_stress. Later rounds will likely keep these mechanics and add more.
- **Score:** the first goal is a safely positive round. Ship the A + B-flat version early as the final submission (expected about +150k), then try to beat it with C, D and the volatile quoting.
