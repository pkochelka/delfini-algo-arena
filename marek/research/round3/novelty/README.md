# Round 3: what is new compared with round 2 (novelty agent)

Tape: `data/round3.csv`. Written 2026-11-08. I ran the same battery of statistics on round 2 (and on round 1's S as a
control) and followed up every material difference. Ownership of the details is on the board (`../NOTES.md`): the ladder
belongs to makers (placement, sizes, cancels) and flow (arrival timing), and the impact decay belongs to s_dynamics.

**Summary:** round 3 is round 2 with two new mechanisms and some changed parameters.
1. **A new "ladder" maker.** It inserts bursts of 2–6 orders exactly 2.000 ms apart, one tick apart, at 2 to 7 ticks
   behind the same-side touch. It is about 5.9 % of maker inserts and is far more active in the volatile regime.
2. **Transient impact.** About half of each taker's impact on S decays, with a time constant of ≈ 18 s (volatile) and
   ≈ 30–45 s (calm). Round 2's impact was fully permanent.

Takers are still coin flips. There is no informed flow, no trend, no seasonality and no new clock apart from the ladder's.

## Ranked differences, round 3 vs round 2

| # | Difference | Effect size | Evidence | Confidence | Kind |
| --- | --- | --- | --- | --- | --- |
| 1 | **Ladder participant** | 69k inserts (5.9 % of makers); 23.8k bursts. Calm ≈ 1.2 bursts/s with rungs of ≈ 150 lots; volatile ≈ 10.6 /s with rungs of ≈ 33 lots | Exact 2000 µs gaps between maker inserts: 45,139 in r3, none in excess in r2. 99.8 % same side, 95 % step +1 tick outward, rungs at touch+2…+7 and never deeper | Certain | New mechanism |
| 2 | **Transient impact** | G(0.5 s) → G(120 s): calm 0.0166 → 0.0063 (≈ 40 % left), volatile 0.062 → 0.033 (52 % left) | Distributed-lag regression of ΔS on signed filled^0.62 (exact S): flat in r2. Price-only test: past→future β on S is −0.14 at 5 s, −0.22 at 10 s, −0.24 at 20 s and −0.31 for 5 s→30 s; r2 and r1 give ≈ 0 | High | New mechanism |
| 3 | Volatile regime occupancy doubled | 29 % of time (r2 15 %). Flow: dwell exit rates 0.0065 /s calm and 0.0166 /s volatile | Taker and maker HMMs (flow) | High | Parameter change |
| 4 | Longer maker lifetimes and larger maker sizes overall | Cancel-life p99 4.9 s (r2 2.0 s); maker log-size mean 2.83 (r2 2.71); maker rate 196 /s (r2 178) | All of it is the ladder (bursts plus its hidden one-rung refills). Non-burst makers run at ≈ 186 /s and look like r2 | High | Consequence of #1 |
| 5 | Immediate impact constants | Calm c ≈ 0.0166 (r2 0.0148); volatile ≈ 0.062 (r2 0.069) | Propagator at 0.2–0.5 s on S v0 | Medium; s_dynamics refits | Parameter change |
| 6 | Smaller price range | 9732–10160 (r2 9444–10017) | Follows from #2: half of all impact washes out | Medium | Consequence |
| 7 | Short-lag positive autocorrelation of S | 1 s→1 s β = +0.09 ± 0.02 on S v0 and +0.15 on the S proxy (r2 S ≈ 0.00) | Not explained by impact: the 0.1 s propagator shows no ramp-up | Low; may be an artefact of S smoothing or of ladder-driven quotes | Unexplained |
| 8 | Volatile maker offset slightly wider | Non-ladder D q90 3.9 (r2 3.5). s_dynamics: 0.69 + Exp(1.06) vs 0.70 + Exp(0.90) | | Low to medium | Parameter change or ladder contamination |

### 1. The ladder (details on the board; makers and flow own the model)
- **Signature:** a burst of maker inserts on one side, each exactly 2000 µs after the previous one, at successive ticks going
  away from the touch. Rung offsets from the same-side touch run 2 to 7. A burst starts at touch+2 in ≈ 60 % of cases,
  otherwise at 3–6 and runs out to 7. Size is almost constant within a burst: the sd of log size is 0.15 within a burst
  and 0.72 between bursts.
- **It tracks the price; it is not spoofing.** Bursts that start at touch+2 follow a mid move away from that side
  (+0.18 / +0.66 / +1.45 ticks over the previous 50 ms / 200 ms / 1 s). Bursts that start deeper follow a move toward
  that side (−0.27 / −0.93 / −1.77).
- **When it cancels:** cancels are bimodal in distance to the touch. 32 % happen at 0–1 ticks, when the price has come to
  the order, and 60 % at ≥ 5 ticks, when the price has run away. Makers measured a hard cancel at more than 7 ticks
  behind (≈ 45 /s).
- **Fills:** 7 % of ladder orders fill in calm and 24 % in volatile.
- **Ladder fills move S like any other fill.** Regressing each taker's immediate kick on total filled^0.62 gives
  R² 0.885 in volatile and 0.449 in calm. Using only the volume filled against regular makers gives 0.683 and 0.341.
- **One-rung refills look like ordinary makers.** The 2 ms test cannot see them, so the "regular" maker stream in calm
  still has a long lifetime tail (p99 4.4 s) and an excess of 90–320-lot orders.

### 2. Transient impact (s_dynamics owns the exact form)
- **Propagator on exact S** (`propag_s.py`, `propag2_s.py`, `propag3_s.py`): impact is fully in place within 0.2 s, with
  no ramp-up, and then decays smoothly.

  | Lag | 0.5 s | 2 s | 5 s | 10 s | 20 s | 30 s | 60 s | 90 s | 120 s | 240 s |
  | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
  | Calm | .0166 | .0162 | .0147 | .0126 | .0109 | .0100 | .0089 | .0078 | .0070 | .0063 |
  | Volatile | .0623 | .0598 | .0550 | .0484 | .0420 | .0379 | .0343 | .0324 | .0324 | – |
  | r2 calm | .0148 | .0147 | .0148 | .0149 | .0148 | .0152 | .0145 | | | |

  Fit for G(τ) = c·[p + (1 − p)·e^(−τ/T)]:
  - Volatile: c = 0.062, p = 0.52, T = 18 s.
  - Calm: c ≈ 0.015, p ≈ 0.45, T ≈ 35–45 s. Less precise: SE ≈ 0.0015 at long lags.

  The decay may run on an event clock rather than in seconds, which would explain why calm decays more slowly. This is
  not tested.
- **The residual also reverts.** The part of ΔS that fills do not explain (ξ plus anything else) gives β = −0.10 for
  10 s→10 s, −0.15 for 20 s→20 s and −0.21 for 5 s→30 s, so ξ is probably transient too.
- **Shock jumps do not revert.** The future move regressed on J gives ≈ 0 from 10 s to 120 s. So S is not a global OU
  process. A fitting form is S = F + Y: shocks and a fraction p of each kick go into F, and the rest goes into Y, which
  decays.
- **Controls:** r2's true S and r1's true S show no reversion in the same tests. The S proxy, which uses no model (mid of
  regular-maker insert bounds), gives the same numbers as S v0.
- **Caveat:** S v0 was filtered with a random-walk (permanent-impact) prior. The decay is visible anyway, but the filter
  should get a decaying state.

## Tested and unchanged vs round 2
- **Takers:**
  - Log-normal size (4.09, 1.40), and the tail above 5k lots matches the log-normal.
  - Side 50/50 and sign ACF ≈ 0.
  - Rates 7.6–7.9 /s calm and 11.6–12 /s volatile.
  - Limit rule vs S: calm round(S ± 1) for 93.6 % plus a tail; volatile S ± 2.55 for 91 %.
  - No chasing: the correlation of taker sign with the past mid or S move is below 0.01 at every lag from 0.05 s to
    15 s (`chase.py`).
  - No informed flow: the signed response is all impact.
  - No sniping: P(taker size equals the opposite touch quantity) matches chance.
- **Non-ladder makers:**
  - Rate ≈ 186–188 /s, the same in both regimes.
  - The size shape matches r2 (makers).
  - Calm D matches r2.
  - No iceberg refills: re-insert at the same price after a full fill matches the shifted baseline, and same size is
    about 1 % in both rounds (`iceberg.py`).
- **Timing:**
  - No exact-µs gaps other than the ladder's 2000 µs (scanned taker→taker, maker→taker, taker→maker, taker→cancel,
    cancel→cancel and cancel→maker).
  - The FFT of 0.1 s counts has no short-period peaks; only low-frequency regime and shock power.
  - Inter-arrival CV ≈ 1.05.
  - Order ids are sequential with no gaps.
- **Seasonality:** no trend in calm taker rate, non-ladder maker rate or taker size across the tape (5-minute bins,
  p > 0.2).
- **Prices:** maker and taker prices mod 5 and mod 10 are uniform, and so is S mod 10/50/100. There is no round-number
  pull.
- **Shocks:**
  - 27 droughts, the same as r2 (flow).
  - Jump sd 48 (r2 42; n = 27, not significant).
  - No pre-shock leak: |corr| < 0.26 between J and drought taker or maker imbalance or the S move before the shock.
  - Jumps do not revert.
- **Volatile episodes:** nothing coincides with them beyond the regime itself. The S change over a segment follows
  decayed impact with slope 0.042, and there is no drift.
- **Not testable on this tape:** a participant that reacts to our quotes. The simulator replays a fixed CSV, so nothing in
  the tape can react to us. The ladder anchors to the touch, so in a live generator our quotes *would* move it.

## What it means for trading
- **Fade order flow.**
  - Keep a transient stock Y(t) = Σ 0.5·c_regime·sign·filled^0.62·e^(−(t − tᵢ)/T), with T = 18 s in volatile and
    ≈ 35 s in calm.
  - On exact S, Y predicts the future move: corr(Y, ΔS) = −0.18 at 5 s, −0.21 at 10 s and −0.23 at 30 s, with slopes
    of −0.24, −0.37 and −0.53 (`ypred.py`). In r2 the same predictor gives ≈ 0.
  - A market maker should skew its quotes against recent fill flow, and an S tracker should add the decay. Expect about
    half of a one-sided sweep to come back within 30–60 s.
- **Track S with a decay term.** A pure random-walk S tracker (as in r2) over-shoots after bursts of flow.
- **Liquidity 2–7 ticks deep.**
  - The ladder adds liquidity at touch+2 to touch+7, which is large in calm (≈ 150 lots per rung) and frequent in volatile.
  - It sits deep and is pulled when the touch gets within ≈ 1 tick or moves more than 7 away. So it rarely trades with
    us, but it cushions sweeps: aggressive takers fill more than in r2, which matters for stale-quote sniping.
  - The ladder disappears in droughts and comes back at ≈ 6 bursts/s after the jump (flow).
- **Regime detector.** Volatile time is ≈ 29 % (r2 15 %). The ladder's 2 ms bursts at > 5 /s make a quick extra
  detector, as does the share of maker inserts in the 25–55-lot band (makers).

## Files
- **Battery:** `battery1.py` (overall diff table), `b2.py`–`b7.py` (lifetime/offset breakdowns), `takers.py`, `tklim.py`,
  `reglife.py`, `season.py`, `periodic.py`, `iceberg.py`, `exactdt.py`, `leaks.py`, `episodes.py`.
- **Ladder:** `layer.py` (exact-dt scan), `chains.py` → `cache/round3_makers_chains.parquet` (cid, cidx, chain per maker
  insert), `c2.py`–`c7.py`.
- **Price dynamics:** `sproxy.py` (S proxy), `vr*.py`, `revert*.py`, `shockrev.py`, `propag*.py` (impact propagator),
  `informed.py`, `chase.py`, `ladderfill.py`, `ypred.py`. `kick*.py` showed that per-fill residuals read off s_hat are
  smeared by the smoother (the same happens in r2), so ignore them.
- **Caches** (`cache/`, built by `prep.py` and `prep2.py`): orders, cancels, trades, top of book per event, a 0.1 s grid,
  and S on the grid (`*_strue.parquet`, from each round's `s_path.parquet`).

## Still open
- The exact decay kernel: exponential or power law, a calendar or event clock, whether ξ decays the same way, and the
  calm p and T. This needs the S filter refitted with a decaying state (s_dynamics).
- Item #7, the +0.09 1 s autocorrelation of S. It is either real short-term momentum that is not tied to fills, or a
  smoothing artefact. Recheck on the refitted S.
