# Round 3: the generator, reverse-engineered

Tape: `data/round3.csv`, 6000 s, 1,234,596 inserts (56,274 IOC takers), 1,178,209 cancels. Written 2026-11-08 from four
parallel agents that coordinated on the board [NOTES.md](NOTES.md) (brief: [BRIEF.md](BRIEF.md)). Their reports hold the
evidence, scripts and CSVs:
- [s_dynamics/README.md](s_dynamics/README.md): the hidden price S, the transient impact law, and regime switch times.
  Exact S per insert is in `s_path.parquet` (needs `uv run --with pyarrow`); the model is in `model_params.json`.
- [flow/README.md](flow/README.md): arrivals, regimes, the shock schedule (`shock_events.csv`) and the ladder event
  table (`ladder_events.csv`).
- [makers/README.md](makers/README.md): base-maker placement and cancels, the ladder maker's rules, and the book.
- [novelty/README.md](novelty/README.md): a round 2 vs round 3 diff of many statistics. This is where both new mechanisms
  were first found.

**Round 3 is the round-2 generator ([../round2/round2_generator.md](../round2/round2_generator.md)) with two additions
and a retuned regime:**
1. **Transient impact.** About half of each taker's kick on S decays exponentially, with T ≈ 14–17 s. In round 2 all of
   it was permanent. Shock jumps do not decay.
2. **A new "ladder" maker.** About four agents fill price levels 2–7 ticks behind the same-side touch in bursts of
   orders exactly 2.000 ms apart. Each burst has one size: about 150 lots in calm and about 34 in volatile.
3. **The calm/volatile chain switches more often.** Volatile is now 28 % of the time (round 2: 15 %).

Takers, the base makers, the shock schedule and the size laws are round 2's unchanged. There are no informed takers, no
momentum, no diffusion, no Hawkes excitation and no seasonality. The lead's "episodes with more and bigger maker inserts"
are the volatile regime, made visible by the ladder; they are not a third state.

## The model

| Part | Calm (72 % of time) | Volatile (28 %) |
| --- | --- | --- |
| Regime dwell | mean 157 s, exponential (round 2: 247) | mean 60 s, exponential (round 2: 43) |
| Kick per *filled* taker | k = sign · **0.0168** · filled^**0.616** + ξ | sign · **0.062** · filled^**0.63** + ξ |
| Noise ξ per filled taker | N(0, 0.130) | N(0, 0.26) |
| **Kick decay (new)** | K(τ) = p + (1−p)·e^(−τ/T), clock time: **p 0.50, T 16.7 s** | **p 0.55, T 13.6 s** |
| Unfilled IOC | no impact | no impact |
| Diffusion of S between trades | 0 | 0 (weak Laplace(0.92) jumps at 0.18 /s, as in round 2, low confidence) |
| Taker rate per side | 4.0 /s | 6.0 /s |
| Taker limit, regular | round(S ± 1), ≈ 95 % | round(S ± 2.55), ≈ 90 % |
| Taker limit, aggressive tail | 2.5 %, a further Exp(3.4) | 4.6 %, a further Exp(5.9) |
| Base maker rate | 180 /s, side 50/50 | 180 /s |
| Base maker offset D | **0.049 + Exp(0.476)**, the round-1 law | **0.673 + Exp(1.00)** (makers agent: 0.57 + Exp(1.05)) |
| Base maker price | bid = min(⌊S − D⌋, ask − 1), ask = max(⌈S + D⌉, bid + 1) | same |
| Base cancel decision hazard | 1.7 + β · max(0, adverse S move) /s, β ≈ 3.4 (round 2: 6.1, see open items) | ≈ 1.7 + β·mv |
| Decision → cancel row | Exp(470 /s) + Exp(52 /s), mean ≈ 21 ms. Round 2's "5 ms dead time" was the 2 ms stage. | same |
| **Ladder inserts (new)** | ≈ 8.5 /s (5.5 singles + 1.1 multi-rung bursts /s) | ≈ 42 /s (8.2 singles + 11.2 bursts /s) |
| **Ladder burst size B (new)** | lognormal(5.02, 0.34), median ≈ 152 | lognormal(3.52, 0.31), median ≈ 34 |

Sizes: base makers round(lognormal(2.661, 1.800)) and takers round(lognormal(4.09, 1.40)), both with a minimum of 1.

**Transient impact, written as a state model.** S = F + Y. A kick k adds p·k to F (permanent) and (1−p)·k to Y, and Y
decays as dY = −Y/T dt. The two regimes' p and T are close; one p ≈ 0.5 and T ≈ 15 s is not excluded. ξ probably decays
with the same kernel (medium confidence: the residual variogram falls ×0.4 at long horizons, as on a synthetic tape where
ξ decays; round 2 stays flat). Shock jumps go straight into F.

### The ladder maker

| Item | Value |
| --- | --- |
| Signature | maker inserts exactly 2.000 ms apart, one side (99.8 %), each one tick further from the touch. 45k exact-2000 µs links; round 2 has none. 23.9k bursts of length 2–6 (12.1k / 6.3k / 2.8k / 1.5k / 1.2k), plus inferred single-order bursts. |
| Who | about 4 independent agents (medium confidence). No level ever holds more than 4 live ladder orders per side, so at most 24 per side. An agent only places at levels it doesn't already hold, nearest first. |
| Band | 2–7 ticks behind the same-side **touch** at insert time, not referenced to S. 61 % of bursts start at touch−2. |
| Rung size | round(B · U(0.75, 1.25)), one B per burst, independent across bursts |
| Trigger | not periodic and not a fixed latency. Rate rises with recent price moves in either direction (calm: 0.38 /s per side when S is still, about 7 /s after a move of more than 3 ticks), and the regime multiplies it 5–7×. The exact wake rule is open. |
| Cancels | inside the band ≈ 0.02–0.2 /s. Beyond touch−7: cancelled at once, subject to the delay above. Near the touch (touch, touch−1): about 2.5–3 /s, with no β·mv term. |
| Lifetime / filled | calm p50 2.8 s, p90 10.8 s, 6 % filled; volatile p50 0.6 s, 24 % filled |
| Shocks | stops completely during droughts, and no ladder order was on the stale side at any of the 27 jumps |
| Impact | fills against ladder orders move S like any other fill |

**Shocks**, on their own clock and independent of the regime (5 of 27 fall in volatile against 7.6 expected):

| Item | Value |
| --- | --- |
| Count and spacing | 27 in 6000 s. First onset ≈ 254 s; gaps 145–293 s, mean 219. Uniform(150, 300) fits (p 0.46). Indistinguishable from round 2 (KS p 0.73). |
| Drought | 3.03 ± 0.05 s. Base maker inserts fall to 0.29× (53.7 /s), the ladder stops, and the taker rate stays normal. |
| Jump | S += J at onset + 3.03 s. sd 48 (round 2: 43), mean \|J\| 33. 6 of 27 have \|J\| < 3, so there may be a "no-jump" class. Normal, Laplace and mixture are not separated. Jumps do not revert, and no feature in the 10–60 s before onset predicts J (\|corr\| ≤ 0.27). |
| Taker burst | 0–0.2 s: about 4 extra takers on the jump side, first a median 77 ms after the jump, hitting the stale book. From +0.2 s: symmetric, about 22 /s per side, falling linearly to baseline by +3 s. The burst size does not depend on \|J\|. The same pattern holds on round 2. |

## Validation

- **S fit (final run):**
  - Calm: 84 of 789k non-ladder maker inserts fall below d0 (0.011 %); 5 are inconsistent with the 99 % interval.
  - Volatile: 188 of 315k.
  - Takers: 94.7 % (calm) and 90.7 % (volatile) sit in the regular band.
  - Posterior sd median: 0.062 tick (calm), 0.107 (volatile).
  - Ladder orders are excluded from the S likelihood, with a 2.6 % uniform outlier term for the single-rung ones the
    2 ms test can't see.
- **Decay method checks:**
  - It prefers permanent impact on round 1 (by 84 nats) and round 2 (by 27 nats).
  - It recovers c, e, p, T and ξ on a synthetic round-3 tape with known truth (`strategy/r3_synth.py`, seed 1). For
    example, calm p comes out 0.582 against a true 0.583.
  - Four independent methods (the impact propagator on S, price-only reversal regressions, the residual variance, and the
    HMM likelihood) all find decay on round 3 and none on round 2.
- **Regimes:** the flow agent's segmentation (takers, ladders, maker prices) and the s_dynamics segmentation (S
  likelihood) find the same 28 volatile episodes, with a median difference of 0.3 s. Ladder rate, rung size and taker
  rate all flip in the same 1 s bin. Exact times: `s_dynamics/vol_segments.csv` (±10–20 ms).
- **Base makers:** counts per size band match 180·P(band) to ±0.3 /s outside the ladder's size bands. Lifetimes are
  within 0.02 s of round 2's.

## Still open

- **The ladder wake rule.** When an agent decides to fill its gaps (fewer than 4 agents refill a gap within 1 s), and
  the near-touch cancel law. A full simulation of the ladder against the tape has not been done.
- **Whether ξ decays.** A direct likelihood test needs a 2-D filter.
- **Whether p and T differ by regime**, and whether calm β really fell from 6.1 to 3.4. The β drop holds on two S
  versions, but it may be an artefact of the transient impact.
- **The calm/volatile c ratio** is 3.7 (round 2: 5.2). The exact volatile D shape is unsettled: 0.67 + Exp(1.00), or
  Gamma k ≈ 3.
- **The shock jump law** (normal, Laplace, or a mixture with a no-jump class) and the exact gap law. There are only 27
  shocks.
- **Lag-1 autocorrelation of 1 s S changes, ≈ +0.09** (round 2: 0.00), low confidence. The novelty agent measured it
  before the final decaying-impact refit, so it may be an artefact of the S smoothing. Recheck it on the final
  `s_path.parquet`.

## What it means for trading

- **Filter out the ladder before estimating S.** This is the biggest effect. A round-2 S filter reads ladder rungs, which
  sit 2–7 ticks off the touch, as wide base-maker quotes: they drag S and P(volatile). The strategy session found
  shockgen_best drops from 772k on round 2 to 451k on round 3. A ladder filter brings it to about 750k (see
  [FINDINGS.md](FINDINGS.md) and `strategy/`). The filter drops rung 2+ of every exact-2000 µs chain, holds any insert
  ≥ 2 behind the touch for 2.01 ms, and raises the outlier share to absorb single rungs.
- **Track S with a decaying term**, F + Y with Y decaying at T ≈ 15 s. The gain is small (≈ +5k). Skewing quotes on
  the transient stock lost money in every test (−40k to −480k). The reversal is real, about −0.2 correlation at
  10–30 s, but slow relative to quote life.
- **Volatile handling matters twice as much.** Volatile is now 28 % of the time. Cheap tells that flip within 1 s are
  exact 2 ms insert pairs (about 11 bursts/s in volatile against 1 in calm) and the share of 20–70-lot maker inserts.
- **Shocks:** the round-2 playbook carries over. The drought gives a 3 s warning, and "no ladder starts" adds a
  confirmation. Have nothing resting on the stale side for the first 0.2 s after the jump, then quote into the
  symmetric burst.
- **Depth 2–7 ticks behind the touch is mostly sticky ladder size.** Our queue position there is worse than the raw
  depth suggests. A ladder rung left at the touch after a move is cancelled slowly and can be picked off, but that is a
  small add-on.
