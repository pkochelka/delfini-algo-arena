# Round 4: the generator, reverse-engineered

Tape: `data/round4.csv`, 6000 s, 1,228,523 inserts (58,445 IOC takers), 1,169,997 cancels. Written 2026-11-08 from four
parallel agents that coordinated on the board [NOTES.md](NOTES.md) (brief and lead recon: [BRIEF.md](BRIEF.md), scripts
in `lead/`). Their reports hold the evidence, scripts and CSVs:
- [s_dynamics/README.md](s_dynamics/README.md): hidden price S with the new velocity term, impact law, regimes, shocks.
  Exact S per insert: `s_path.parquet` (`uv run --with pyarrow`); model: `model_params.json`; switch times: `vol_segments.csv`.
- [flow/README.md](flow/README.md): taker side law, rates, limits, shock schedule (`shock_events.csv`), regimes
  (`regimes.csv`), ladder table (`ladder_events.csv`).
- [makers/README.md](makers/README.md): the 15-lot agent (`r15_episodes.csv`), ladder (now 3 agents), base makers.
  Per-insert class labels for everything: `makers/cache/order_class.parquet` (base / ladder / r15 / regular taker / TWAP).
- [novelty/README.md](novelty/README.md): ranked round 3 → round 4 diff, the TWAP taker (`twap_episodes.csv`), and the
  causal leak regressions.

**Round 4 is the round-3 generator ([../round3/round3_generator.md](../round3/round3_generator.md)) with five changes:**
1. **S has a velocity V.** S drifts between trades. Volatile: Gaussian V, sd ≈ 3.1 tick/s, persistence ≈ 2–3 s. Calm: weak,
   sd ≈ 0.4 tick/s. Fills do not feed V.
2. **Takers pick their side from S − mid:** P(buy) = Φ((S − mid)/4.42), same in both regimes. In round 3 the side was a
   coin flip. This one rule is the whole taker sign persistence (ACF 0.08) and the apparent "chasing".
3. **A TWAP taker:** 37 one-sided episodes of 22–58 IOCs of exactly 160 lots, exactly 0.500000 s apart.
4. **A 15-lot "informed" joiner:** 31 one-sided episodes (calm only) in which it keeps one 15-lot order at its own touch,
   re-posting 1 µs after each fill/cancel, while S drifts its way at ≈ 1 tick/s.
5. **Ladder retuned:** 3 agents instead of 4 (≤ 3 live per level), rungs ≈ 4/3 bigger. Shock jumps look bigger (sd 63 vs 48).

Together, 1 and 2 make the tape super-diffusive. V moves S away from the book, the side law sends takers the way of
S − mid, and their kicks extend the move. At 300 s the variance splits into drift 49 %, impact 18 % and covariance 33 %.
The price range is 9178–10033 (round 3: 9758–10138).

## The model

S = F + Y + ∫V dt. A filled taker's kick k = sign · c · filled^e + ξ adds p·k to F and (1−p)·k to Y. Y decays as
dY = −Y/T dt. Shock jumps go into F.

| Part | Calm (70 % of time) | Volatile (30 %) |
| --- | --- | --- |
| Regime dwell | mean 135 s | mean 56 s (possibly gamma-like rather than exponential, n = 32) |
| Kick per filled taker | 0.0145 · filled^0.62 + ξ, ξ ~ N(0, 0.10) | 0.048 · filled^0.65 + ξ, ξ ~ N(0, 0.22) |
| Kick decay | p 0.56, T 14 s (round 3: 0.50, 16.7) | p 0.61, T 18 s (round 3: 0.55, 13.6) |
| **Velocity V (new)** | sd ≈ 0.4 tick/s, low confidence (hard to tell apart from ξ) | **sd ≈ 3.1 tick/s, persistence ≈ 2–3 s (1.6–3.1 by method), Gaussian, exogenous**; possibly mean −0.37 tick/s (t −2.4, low) |
| V vs regime switch | V seems to lead the quote/kick switch by ≈ 0.75 s at both ends (medium) | |
| Taker rate per side | 3.9–4.0 /s | 5.9–6.0 /s |
| **Taker side (new)** | **P(buy) = Φ((S − mid)/4.42) ≈ logistic(0.37·(S − mid))**, mid = plain book mid | same law; stronger effect only because \|S − mid\| is wider (sd 1.9 vs 0.5) |
| Taker limit, regular | round(S ± 1), 94.6 % | round(S ± 2.55), 87.2 % |
| Taker limit, tail | 5.3 %, excess mean 3.2 | 12.4 %, excess mean 6.5 (tail takers lean harder: σ ≈ 2) |
| Taker size | round(lognormal(4.094, 1.400)) | same |
| Base maker rate, size | 180 /s; round(lognormal(2.66, 1.80)) | same |
| Base maker offset D | 0.03–0.07 + Exp(0.45–0.47) | 0.676 + Exp(1.00) |
| Base cancel hazard | as round 3 bin for bin (β fit 2.7 vs 3.4, probably S-filter artefact) | same |
| Ladder burst size B | lognormal(5.33, 0.33), median ≈ 205 (round 3: 5.02) | lognormal(3.82, 0.27), median ≈ 46 (round 3: 3.52) |

Given S − mid, taker signs are independent (residual ACF ≈ 0 at lags 1–4). Past signs or past moves add 5–20 nats
against ≈ 1,000 for S − mid. Rejected: hidden meta-order state, trend following (no chasing of shock jumps after +0.2 s),
an extra arbitrage class (the total rate does not depend on S − mid), and look-ahead (lagging S by 0.2 s kills the fit).

### The TWAP taker (new)

| Item | Value |
| --- | --- |
| Signature | IOC, exactly 160 lots, exactly 500,000 µs apart, one side per episode. Size spike at 160: 1,556 vs 83 expected |
| Episodes | 37; children n ~ U(22..58) (KS p 0.74), so 10.5–28.5 s; ≈ 12 % of the tape. Does not follow the side law |
| Price | each child takes 160 at touch or touch + 1; S moves +14.4 ± 5.2 ticks its way over the episode (+0.37 per child = plain impact) |
| After | ≈ 8 ticks revert within 10 s of the last child (the transient part of the impact) |
| Information | none: it is uninformed and is not the source of the sign ACF |

### The 15-lot agent (new)

| Item | Value |
| --- | --- |
| Signature | 755 maker orders of exactly 15 lots in 43 instances, 31 episodes (12 buy, 19 sell), median 8.2 s, starts ≈ every 195 s, all in calm |
| Rule | holds one 15-lot order at its own-side touch (754 / 755). Re-posts exactly 1 µs after its own fill or cancel; after a partial fill it re-posts the remainder. Cancels only when the touch improves past it (350 / 352), with the base-maker cancel delay (mean 23 ms) |
| Stop | 39 / 43 instances stop after a full fill with no re-post; total filled 15·n, n ≈ U(1..18) (medium confidence) |
| The 1 µs pairs | it is 364 of the 382 taker→insert 1 µs pairs and all of the round-4 excess of cancel→insert 1 µs pairs (348 of 690) |
| **Drift** | **S moves its way by +6.9 ticks per instance (42 / 43 positive), ≈ 1 tick/s, on the clock, not on its fills.** The drift starts with its first order and stops at its last fill, then 3–5 ticks come back. Open: does it cause the drift or announce it? |

### Ladder maker

As round 3 (2.000 ms rungs, one side, touch−2 … touch−7, rung size round(B·U(0.75, 1.25)), regime-driven rate, 21.8k
ladders), but **3 agents** (≤ 3 live orders per level) with bursts ≈ 4/3 bigger, so depth per level is about the same.

### Shocks

28 droughts of 3.02 s (maker inserts at ≈ 0.29×, ladder stops). Then S += J, J ~ N(0, 63) (round 3: sd ≈ 48; p ≈ 0.1,
only 28 jumps; KS normal p 0.99). No reversion and no no-jump class. First onset 167.5 s; gaps as before (≈ U(150, 300)).
Jump head: ≈ 4.5 extra jump-side takers in 0–0.2 s, a separate component from the side law, as in round 3. The
post-jump symmetric burst may now grow with |J| (low confidence).

## Validation

- **S filter:** round 3's HMM with ladder chains (66k inserts) and the 15-lot orders left out. A Kalman-smoothed velocity
  is added to each step, iterated three times. Log-likelihood went from −1,098,290 (round-3 parameters) to −1,071,071.
  The velocity uses data from both sides in time, so this gain overstates the evidence. On the path itself, a Kalman
  filter with V gains ≈ 9,800 nats over one without. Without V the volatile impact looks almost permanent, because the
  drift leaks into it.
- **Independent confirmation of V:** novelty took the fill impact out of exact S. The residual's autocorrelation at
  0.5 s is 0.56 in volatile (≈ 0 in round 3), and its variance in 0.5 s windows with no fills is 1.75 (round 3: 0.07).
- **Side law:** stable across regime, post-jump time, spread, size and thirds of the tape (probit σ 4.1–4.7). On round 3's
  S the same fit finds a slope ≈ 0. It also explains why an unfilled IOC is followed by an opposite-side taker
  (corr −0.54 in volatile).
- **Base makers, sizes, lifetimes, timing clocks, order ids, seasonality:** unchanged. There are no icebergs, no sniping
  and no pre-shock leak.

## Still open

- **The closed loop.** Side law + impact + V + book lag has not been simulated to reproduce the sign ACF 0.079 and the
  5 s imbalance-to-move correlation. An open-loop simulation gives about half, which is expected without the feedback.
- **V's law:** OU or random jumps; persistence 1.6–3.1 s by method. Whether calm V is really the 15-lot agent's drift
  episodes. The causal velocity estimator was not checked on a synthetic tape.
- **The 15-lot agent:** does it cause the drift or announce it? Its start, side and stop rules rest on 31 episodes.
- **The volatile mean drift** (−0.37 tick/s, low confidence) and the volatile dwell law.

## What it means for trading

This round's edge is **short-horizon predictability of S**. Round 3 punished leaning (impact decays); round 4 rewards
leaning with momentum over 1–5 s.

1. **Quote around S + V·h, not the book mid.** A quote that lags S gets a Φ(lag/4.42) share of the takers on the stale
   side. In volatile, lots taken from the stale side gain the taker ≈ 22 S-ticks per lot by +5 s.
   - With a causal S filter, S − mid predicts the next 1 s mid move with R² 0.40 in volatile and 0.20 in calm, and the
     next taker's side with corr 0.29.
   - The causal V predicts volatile S moves with R² 0.30 at 1 s and 0.33 at 2 s (≈ 1.6×V and 2.9×V). Calm R² is ≈ 0.03.
   - Rule of thumb (volatile): a 1-tick move in 0.5 s is followed by ≈ 1.8 more over 5 s.
   - Without S, tape-only tells reach R² 0.33 in volatile: 0.25 s taker imbalance, the side of the last unfilled IOC,
     and the past mid move.
2. **Upgrade the r3 S filter with a velocity state**, and keep the ladder filter. It must now also drop the 15-lot orders
   (exactly 15 lots at the own touch, re-posted 1 µs after a fill or cancel) from the maker constraints.
3. **The 15-lot agent is an informed-flow tell, not liquidity.** Its first 1 µs re-post comes a median 0.57 s into the
   episode. From there the mid moves +5.2 ticks in 5 s and +8.4 by the end (worst case −1). Don't quote against it;
   lean its way. After it stops, expect 3–5 ticks back.
4. **The TWAP is free information.** It is easy to detect: 160 lots, then 160 lots again exactly 0.5 s later. Its child
   count is ≈ U(22..58). Lean with it while it runs (+0.37 tick per child), don't sell into a buy TWAP at the touch, and
   fade ≈ 8 ticks after the last child.
5. **Volatile handling still matters most** (30 % of time; V is mostly a volatile phenomenon). A fast calm move with no
   fills is an early warning of a volatile episode (V leads the switch by ≈ 0.75 s).
6. **Shocks:** the round 2–3 playbook carries over: 3 s drought warning, nothing on the stale side for the first 0.2 s.
   Jumps are bigger (sd ≈ 63), so the exposure cap at onset matters more.
