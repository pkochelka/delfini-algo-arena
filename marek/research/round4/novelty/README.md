# Round 4: what is new compared with round 3 (novelty agent)

Tape: `data/round4.csv`. Written 2026-11-08. I reran round 3's battery (`../../round3/novelty/`) on round 3 vs round 4,
then followed up every material difference. Ownership is on the board (`../NOTES.md`): S and its drift belong to
s_dynamics, the taker side law and shocks to flow, and the ladder and the 15-lot agent to makers. Exact S is
s_dynamics' v0 `s_path.parquet` (round-3 model, so it has no drift term). Every predictive test below uses the causal
`s_filt` for the past and `s_hat` for the future.

**Summary:** round 4 is round 3 plus three new things, a retuned ladder and a bigger shock jump.
1. **S has its own velocity.** S drifts on clock time, independently of trades. The drift persists for about 2.5–3 s
   and is 5–10× stronger in the volatile regime. This is the "super-diffusion" and the main reason the price range is
   2.3× wider.
2. **Takers now pick their side from S − mid** (flow's finding, which I cross-checked). P(buy) = Φ((S − mid)/4.42). In
   round 3 the side was a coin flip.
3. **Two new agents.** A TWAP taker sends 160-lot IOCs exactly 0.5 s apart, in runs of 22–58. A 15-lot passive joiner
   sits at the touch (makers own it). It shows up while S drifts in its direction.
4. **Ladder rungs are 30–40 % bigger.** Shock jumps have sd 65 against 49 in round 3 (n = 28, weak).

Base makers, taker sizes and limits, regime occupancy, lifetimes, the shock schedule, ids and timing are unchanged.

## Ranked differences, round 4 vs round 3

| # | Difference | Effect size | Evidence | Confidence | Kind |
| --- | --- | --- | --- | --- | --- |
| 1 | **Trade-independent S velocity (momentum)** | Volatile: residual ΔS ACF 0.56 at 0.5 s, decaying with τ ≈ 2.5–3 s; residual VR at 40 s 6.7 (r3 0.39). Calm: ACF ≈ 0.25 out to 3 s, then 0 by 4–5 s (box-like); VR 3.0 (r3 0.50) | Fill-impact regression on exact S, then tests on the residual (`sdrift.py`, `velacf.py`). The residual variance in bins with no fills is 1.75 in volatile (r3 0.07) (`xiramp.py`). S moves in 0.5 s windows with no fills: mean \|ΔS\| 1.11 volatile (r3 0.16), and the move continues the past 2 s (corr +0.60; r3 −0.27) (`nofill.py`) | High | New mechanism |
| 2 | **Taker side depends on S − mid** | logit slope 0.37 per tick in both regimes. dLL +1081 in volatile, +178 in calm, using causal s_filt at the previous row. r3: slope 0.00, dLL < 1 | flow's `sidelaw.py`, cross-checked with no own-kick leak in `sidecheck.py`. This produces the taker sign ACF of 0.08 (r3 0.00) and the "chasing" corr +0.21 | High | New mechanism |
| 3 | **TWAP taker** | 1,490 IOCs of 160 lots, 37 episodes, exactly 500,000 µs apart, one side per episode. n ~ U(22..58) (KS p 0.74), so episodes last 10.5–28.5 s; 12 % of the tape | Size spike at 160 (1,556 against 83 expected), the only exact long-range clock (`tksize.py`, `longdt.py`, `twap.py`) | Certain | New agent |
| 4 | **15-lot passive joiner** (makers) | 755 orders, 31 episodes, all in calm. S drifts about 1 tick/s in its side's direction while it is active (+6.7 ticks per window) | Makers' table. My check: residual ACF inside its windows is 0.46 against 0.34 outside, so it explains little of the calm momentum (`r15x.py`) | High | New agent and a tell |
| 5 | Wider price range and larger mid moves | Range 838 (r3 364). 60 s mid sd 55 (r3 34). Volatile 1 s mid ACF +0.54 (r3 +0.19) | `battery1.py`, `acf.py`, `vr.py` | High | Consequence of #1 and #2 |
| 6 | Ladder rung size up | Median B: calm 187 (r3 145), volatile 46 (r3 33). Burst rate and band unchanged. 78 chains of length ≥ 7 (r3 51). Non-chain maker sizes 200–300 up and 110–160 down (single-rung refills) | `ladcmp.py`, `mksize.py` | High | Parameter change |
| 7 | Shock jump sd | 65 (r3 49), n = 28, F-test p ≈ 0.07. Corr of the S move over the 30 s before the shock with J is −0.37 (p ≈ 0.05, probably chance) | `shockrev.py`, `leaks.py` | Low to medium | Parameter change? |
| 8 | Volatile taker limits look wider against S | 87 % in the round(S ± 2.55) band (r3 91 %); q90 of signed limit − S is 4.8 (r3 3.0) | `tklim.py`. Limits still reference current S (`tklag.py`) | Medium | flow owns it; may come from #1 (S moves within the latency) |
| 9 | Volatile starts on integer seconds | 12 of 32 s_dynamics volatile starts lie within 0.05 s of an integer second (3.2 expected). Ends: 2 of 31. r3: 5 of 28 | `vol_segments.csv`. The integer-start segments differ in nothing else (`intstart.py`) | Low; may be an artefact of segmentation seeded on a 1 s grid | Unexplained |

### 1. S velocity (s_dynamics owns the model)
- **Method.** I regress 0.5 s ΔS (exact S v0) on signed filled^0.62, with separate calm and volatile kernels 60 s long.
  - The fill kernel is round 3's: volatile G(0.5 / 2 / 5 / 10 / 30 / 60 s) = .058 / .062 / .064 / .060 / .052 / .055, and
    calm .016 / .016 / .015 / .012 / .012 / .009. There is no ramp-up, so the momentum is not delayed impact.
  - The residual is what is new:

  | Residual ΔS ACF at lag | 0.5 s | 1 | 1.5 | 2 | 3 | 4 | 5 | 7 | 10 |
  | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
  | r4 volatile | .56 | .48 | .40 | .30 | .20 | .15 | .10 | .06 | .04 |
  | r4 calm | .34 | .24 | .28 | .20 | .21 | .06 | −.01 | −.02 | −.04 |
  | r3 volatile | −.09 | −.05 | −.02 | −.01 | .00 | .05 | −.01 | −.01 | .00 |
  | r3 calm | −.05 | −.02 | .03 | .03 | −.02 | .01 | −.04 | −.01 | −.02 |

- **It runs on clock time.**
  - The residual variance per 0.5 s bin, by number of fills in the bin (0 / 1 / 2 / 3): r4 volatile 1.75 / 2.79 / 2.80 /
    1.81, against r3 volatile 0.07 / 0.62 / 0.55 / 0.69. The variance does not depend on fills.
  - corr(#fills_i, resid²_{i+l}) ≈ 0 for every l ≠ 0, so it is not the noise ξ spread over time.
  - Trade-free S velocity in volatile has median 1.3 tick/s and q90 4.8 (r3 0.43 / 1.08). In calm the median is 0.11
    tick/s (r3 0.08), with heavier tails.
- **Shape.**
  - Volatile looks like an OU velocity with τ ≈ 2.5–3 s.
  - Calm looks more like segments of constant velocity a few seconds long. The 15-lot agent's windows (median 6 s,
    ≈ 1 tick/s) are one visible subset of them.
  - Within a volatile segment the cumulative residual shows 5–20 s trends, then a plateau or partial give-back. The net
    residual per segment is \|31\| ticks on average (r3 5.5).
  - At 20–60 s, mid and S moves revert (corr ≈ −0.25 in volatile). That is the round-3 transient impact plus the decay
    of the drift.
- **Takers are not informed about it.** Once the past residual is in the regression, past taker imbalance adds
  nothing in volatile (t ≈ 1). Limits reference current S: the in-band fraction of limit − S(t + δ) peaks at δ = 0 and
  is symmetric.
- **Round 3 open item #7.** On round 3's final S, the raw 1 s → 1 s β is +0.03 in calm and +0.10 in volatile. Round 3
  had at most a trace of momentum; round 4 has the real thing.

### 2. Side law (flow owns it), cross-check
- **Causal check.** The test uses s_filt at the row before each taker, so it has no look-ahead and no own kick.
  - Slope: r4 calm +0.368 (dLL 178), volatile +0.403 (dLL 1081). r3: −0.019 and +0.004 (dLL < 1).
  - With s_hat at the row before or the taker's own row, the slopes are the same within 0.03.
- **What it explains.** It accounts for the taker sign ACF, the chase correlation (+0.21 in volatile at 0.2–0.5 s) and
  "unfilled taker, then an adverse move".
- **It is not the momentum.** Momentum (#1) is present with no fills at all. The two mechanisms reinforce each other:
  S drifts away from the book, takers trade toward S, and their kicks push S further.

### 3. TWAP (makers have the same table)
- **Exact rules.**
  - 160 lots per child, Δt exactly 500,000 µs, with no skipped slots.
  - Duration is exactly (n − 1) · 0.5 s. Episodes start at a random phase and never mix sides. 20 are buys and 17 are
    sells.
  - The limit uses the regular limit rule without the aggressive tail. Signed limit − S runs from 0.4 to 2.5, and 96 %
    of children fill.
- **Uninformed.** In the 5–30 s before the start there is no signed move (+0.7 ± 2.5 and +1.5 ± 7.3). The start is not
  tied to shocks.
- **Price effect.** During the episode S moves +14.4 ± 5.2 ticks in the TWAP direction. That is 0.37 per child, which
  is exactly the calm impact of 160 lots.
  - Signed S move over the next 5 s, by child index: 2–5: +3.3, 10–20: +2.2, 30+: −0.8 (the end is coming).
  - After the last child: −2.5 / −6.8 / −8.4 / −4.3 / +0.7 at +2 / 5 / 10 / 20 / 30 s.
- **It is not the source of the sign ACF.** Without the TWAP children the ACF at lags 1–3 is 0.080 / 0.072 / 0.062.

### 4. Joint causal leak regression (`leakjoint.py`)
Future S (s_hat at t + H) minus s_filt at t, regressed on causal features. Coefficients are in ticks per sd of the
feature, with t-stats in brackets.

| | H | R² | past 0.5 s ΔS | past 0.5–2 s ΔS | S − mid | taker imb 1 s | ladder burst imb 1 s | TWAP side | r15 agent side |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| r4 volatile | 2 s | **0.240** | +2.35 [6] | +3.00 [11] | +3.60 [9] | −0.73 [−2] | +0.25 [1] | ≈ 0 | +0.50 [2] |
| r4 volatile | 5 s | 0.142 | +4.58 [5] | +4.09 [7] | +5.35 [7] | −2.12 [−3] | +0.84 [2] | ≈ 0 | +1.66 [3] |
| r4 calm | 2 s | 0.045 | +0.15 [4] | 0 | ≈ 0 | +0.07 [2] | +0.12 [4] | **+0.35 [11]** | **+0.30 [10]** |
| r4 calm | 5 s | 0.017 | +0.23 [2] | ≈ 0 | ≈ 0 | ≈ 0 | +0.17 [2] | **+0.78 [8]** | +0.51 [6] |
| r3 volatile | 2 s | 0.032 | −0.79 | −0.41 | +1.99 | +0.41 | ≈ 0 | – | – |
| r3 calm | 2 s | 0.002 | ≈ 0 | ≈ 0 | ≈ 0 | +0.10 | ≈ 0 | – | – |

The r15 agent side is the direction of its drift, i.e. its own side. The price-only version: in volatile, the past
1 s mid move predicts the next 5 s mid move with corr +0.36 (r3 +0.04). With causal S, the past 0.5 s move predicts the
next 5 s with β = +1.83, so a 1-tick move is followed by 1.8 more.

## Tested and unchanged vs round 3
- **Rates and sizes.**
  - Maker rate 195 /s (r3 196); taker rate 9.7 /s (9.4; the TWAP adds 0.25).
  - Sides 50/50.
  - Regime taker rates: calm 7.96, volatile 11.87 (r3 7.64 / 11.59).
  - Taker size lognormal (4.12, 1.39), with the tail above 5k as expected. The only spike is 160. Multiples of 5 and 10
    are over-represented only because of it.
  - Maker size shape is the same, apart from the ladder bands. The 15-lot base count is smooth once the agent's orders
    are removed (makers).
- **Regular (non-ladder) makers.**
  - Offset D is the same as r3: calm q10/50/90 0.42 / 1.01 / 1.94, volatile 1.20 / 2.08 / 4.09 (r3 1.17 / 2.02 / 3.90).
  - Lifetimes: calm p50 0.302 s and p99 4.04 (r3 0.305 / 4.43); volatile p50 0.229 (0.241).
  - Fill rate 14.7 % (14.2 %). Never crossed.
- **Ladder.** Burst rate (calm 0.98 /s, volatile 10.3 /s; r3 1.23 / 10.8), band touch+2…+7, one-sided, and the 2.000 ms
  clock are all unchanged. Only the size has changed (#6).
- **Timing.**
  - Exact-µs scan within 20 ms: the only new pairs are taker→maker 1 µs (382, the 15-lot agent) and cancel→maker 1 µs
    (690, of which about 348 are the agent; the rest matches r3's background).
  - The same-size long-range scan up to 10 s finds only the TWAP's 0.5 s clock and the ladder's 2 ms clock.
  - No zero gaps. Inter-arrival CV 1.05. FFT shows no short-period peaks.
  - Order ids are sequential with no gaps.
- **Seasonality.** No trend in calm taker rate, regular maker rate or taker size across the tape (p ≥ 0.47 for
  the rates; taker log-size p 0.03 comes from the TWAP's 160s). Volatile occupancy per 1000 s is 0.15–0.41 (r3 0.20–0.39); overall 0.28–0.30.
- **Prices.** Maker and taker prices mod 5 and mod 10 are uniform, and so is S mod 10/50/100. No round-number pull.
- **No iceberg.** The reinsert excess at the same price within 10 ms of a full fill is +3.6 % (r3 +1.9 %); same-size
  refills are 0.6 % (r3 0.5 %). The 15-lot agent's re-posts account for the difference.
- **No sniping.** P(taker size equals the opposite touch quantity) is at chance (0.0021 against 0.0022).
- **Shocks.** The schedule, drought (3.0 s, about 54 maker inserts/s) and taker burst follow flow and match r3. Jumps do
  not revert (β on J is about 0 out to 60 s). There is no pre-shock leak from drought flows (\|corr\| < 0.18).
- **The spread and touch** are unchanged: mean 1.26, and the spread is 1 tick 81 % of the time.

## Still open
- **The exact drift law** (s_dynamics). Questions: an OU velocity in volatile with τ ≈ 2.5–3 s against piecewise-constant
  trend segments; what calm looks like, given its box-like ACF of about 4 s; whether the velocity is redrawn at regime
  switches; and the drift sd in each regime. The S filter needs a velocity state, and v0 has none. I measured the
  momentum on v0, so the numbers may sharpen once S is refitted.
- Whether the 15-lot agent "sees" the calm drift (it starts with it) or causes it. Its fills move S against it, so the
  drift is not its impact.
- Whether integer-second volatile starts (#9) are real. Recheck once the segmentation has its final refinement.
- The jump sd increase (#7) rests on only 28 jumps.

## What it means for trading
- **Flip round 3's fade into momentum at short horizons.**
  - In volatile, S keeps going: a 1-tick causal S move in 0.5 s is followed by about 1.8 more over 5 s (R² 0.24 at 2 s
    from price-only features).
  - Skew quotes with the recent S move. Pull or widen the side S is moving toward, because it will be run over. Fade
    only after 20–60 s.
  - A round-3 fade (skew against recent flow) should lose here.
- **The book lags S, and takers trade on the lag.**
  - Takers buy when S > mid. Every resting quote on the side S is moving toward is adversely selected, and in volatile
    S runs away from the book at about 1–5 tick/s.
  - Track S with a velocity state, e.g. a Kalman filter with a [level, velocity] state. Centre quotes on S plus
    velocity × the expected hold time, not on S.
- **TWAP is free information.**
  - Two 160-lot IOCs 0.5 s apart on one side mean at least 20 more are coming (n ≥ 22). Expect +0.37 tick of S per
    child, about +0.75 tick/s, in that direction.
  - The end is unknown but uniform on 22..58 children, and after the end about −8 ticks come back over 10 s.
  - Lean quotes in the TWAP direction during the episode, take the other side of the reversal after it, and do not
    sell into a buy TWAP at the touch. Each child takes 160 lots at touch or touch + 1.
- **The 15-lot joiner is a calm-regime drift tell.** A lone 15-lot order joining the touch, re-posted 1 µs after its
  own fill or cancel, means S drifts about 1 tick/s toward that side for about 6 s (t ≈ 10).
- **Volatile handling matters even more than in round 3.** Occupancy is the same (about 29 %), but price risk per
  second is about 2× higher. The regime tells still work: ladder burst rate, 20–70-lot maker share, taker rate. Ladder
  rungs are bigger, so depth at 2–7 ticks is about 35 % deeper.
- **Shocks.** Same playbook. Jumps may be bigger (sd 65), so size inventory limits around the 3 s drought warning for
  that.

## Files
- **Caches** (`cache/`):
  - Round-4 caches are built by `prep.py`, `prep2.py`, `chains.py` and `prep_s.py`. `prep_s.py` makes S on the grid
    from `../s_dynamics/s_path.parquet`.
  - Round-3 caches are symlinks into `../../round3/novelty/cache`.
  - `regproxy.py` is a fallback regime proxy, not used once `../flow/regimes_1s.csv` exists. `common.py` picks the
    regime and shock files automatically.
- **Battery** (copied from round 3 and repointed to r3 vs r4): `battery1.py`, `exactdt.py`, `periodic.py`, `iceberg.py`,
  `season.py`, `acf.py`, `vr.py`, `informed.py`, `takers.py`, `chase.py`, `tklim.py`, `reglife.py`, `leaks.py`,
  `shockrev.py`.
- **New for round 4:**
  - Sizes and clocks: `tksize.py`, `mksize.py`, `longdt.py`, `ladcmp.py`.
  - TWAP: `t160*.py`, `twap*.py`, with the episode table `twap_episodes.csv` and per-taker flags in
    `cache/round4_takers_tw.parquet`.
  - Momentum: `leakpred.py`, `mech.py`, `drift*.py`, `sdrift.py`, `velacf*.py`, `nofill*.py`, `xiramp.py`.
  - Leaks: `informed2.py`, `leakcausal.py`, `leakjoint.py`.
  - Cross-checks: `sidecheck.py` (flow's side law), `r15x.py` (makers' agent), `react15.py`, `tklag.py`,
    `intstart.py`, `mkleak.py`.
