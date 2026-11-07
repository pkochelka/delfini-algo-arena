# Round 2: the generator, reverse-engineered

Tape: `data/round2.csv`, 6000 s, 1,123,282 inserts. Written 2026-11-07. This page combines three reports, which hold the
evidence, scripts and CSVs:
- [s_dynamics/README.md](s_dynamics/README.md): the hidden price S, impact law and regime switch times. Exact S per insert
  is in `s_path.parquet`, and the final fit in `params_v3.json`.
- [flow/README.md](flow/README.md): arrivals, the regime HMM and the shock schedule (`shock_events.csv`).
- [makers/README.md](makers/README.md): placement, cancels and the book in each state.

**Round 2 is the round-1 generator (`strategies/peter/round1_generator.md`) with three additions:**
1. **A two-state Markov regime, calm or volatile.** It switches every price parameter at once.
2. **A new impact law:** filled^0.62 instead of √filled, plus Gaussian noise on each filled taker.
3. **27 scheduled shocks:** a 3 s maker drought, then an exogenous jump in S.

Takers are still coin flips. There are no informed takers, no momentum, no mean reversion, no drift between trades,
no Hawkes excitation and no seasonality. Sizes follow the round-1 lognormals exactly, and the 72k-lot makers are just
that distribution's tail.

## The model

| Part | Calm (85 % of time) | Volatile (15 %) |
| --- | --- | --- |
| Regime dwell | mean 247 s, geometric (exit ≈ 0.004 /s) | mean 43 s (7–229 s, exit ≈ 0.023 /s) |
| Impact per *filled* taker | ΔS = sign · **0.0142** · filled^**0.625** + ξ | sign · **0.0741** · filled^**0.61** + ξ |
| Noise ξ per filled taker | N(0, 0.124) | N(0, ≈0.28) |
| Unfilled IOC | no impact | no impact |
| Diffusion of S between trades | 0 | ≈ 0 |
| Taker rate per side | 4.0 /s | 6.0 /s |
| Taker limit, regular (≈95 % calm, ≈92 % volatile) | round(S ± 1) | round(S ± 2.55) |
| Taker limit, aggressive tail | 2–5 % of takers: a further 0.5 + Exp(≈3) | 4–8 %: a further 0.5 + Exp(≈6) |
| Maker rate | 180 /s, side 50/50 | 180 /s |
| Maker offset D | **0.05 + Exp(0.45)**, the round-1 law | **0.70 + Exp(0.90)**; no quote ever within 0.70 of S |
| Maker price | bid = min(⌊S − D⌋, ask − 1), ask = max(⌈S + D⌉, bid + 1), clipped and never dropped | same |
| Cancel decision hazard | 1.73 + 6.1 · max(0, adverse S move) /s | ≈ 1.9 + (3.4–4.8) · mv |
| Decision → cancel row | ≈ 5 ms dead time + Exp(≈ 50 /s), mean ≈ 25 ms. The row is written even after a fill. | same |

Sizes: makers round(lognormal(2.657, 1.803)), takers round(lognormal(4.10, 1.40)), both with a minimum of 1.

**Shocks** run on their own clock, independent of the regime:

| Item | Value |
| --- | --- |
| Count and spacing | 27 in 6000 s. First onset 270.9 s; gaps 147–296 s, mean 214. Not Poisson. U(150, 300) fits, and so does 150 + Exp(64). |
| Drought | Maker inserts fall to 0.30× (about 54 /s) for exactly 3.0 s. Cancel α ≈ 6–6.6 /s; D wider, mean ≈ 1.9. The touch falls from about 900 to about 115–210 lots. |
| Jump | At onset + 3.00 s: S += J, J ~ N(0, 43) (normal vs Laplace not resolved). Range −87 to +113, mean \|J\| 30. Unpredictable: \|corr\| < 0.3 with every drought feature. |
| Stale book after the jump | 0–2369 lots per shock (8.2k in total). Still resting: 94 % at 5 ms, 75 % at 10 ms, 36 % at 20 ms. Only 2.5 % fills. The first taker in the jump direction arrives a median 70 ms after the jump. |
| Taker burst | +35 /s total for 1.25 s, falling linearly to the baseline rate by +3 s (≈ 80 extra takers), symmetric. |

## Validation

- **Calm:** with the S path from the fitted model, 0 of 908,240 calm maker inserts break the price rule.
- **Volatile:** 18 of 158,381 volatile inserts break it, all within 0.4 s of a regime switch.
- **Takers:** 94.7 % (calm) and 90.9 % (volatile) sit in the regular band; the rest are the aggressive tail.
- **Method check on round 1:** the same pipeline recovers round 1's truth, e = 0.5 and c = 0.0258, and rejects e = 0.6
  and ξ = 0.016.

## Still open

- **How volatile scales the calm parameters.** The ratios are c ×5.2, taker offset ×2.55, ξ ×2.2, and D shifts from
  0.05 + Exp(0.45) to 0.70 + Exp(0.90). No single multiplier fits them all, so the generator probably draws separate
  constants per regime.
- **Volatile β and the decision-to-cancel delay.** Both are rough because S is less precise there.
- **Small residual volatile jumps** (Laplace 0.84 at 0.23 /s). They may be a misfit.
- **The aggressive-tail shape**, the exact exponent (0.60–0.65), and the shock gap and jump laws. There are only 27 shocks.

## What it means for trading

- **Calm:** round 1 carries over unchanged, apart from S. Track S with 0.0142·filled^0.625. The ξ noise means S drifts
  away from a trade-only estimate (sd ≈ 0.12·√(number of fills)), so re-anchor it continuously from the maker-insert
  bounds, as `s_ladder` already does.
- **Volatile:** the instant tell is maker inserts that never come within 0.70 of S. In calm, 74 % of inserts do. The
  1 s spread > 1.5 rule also works, with about 1 s of lag. Impact is about 5× calm, and stale-touch episodes are 13× more
  frequent: 1.4 /s per side, edge 1.3–1.9 ticks.
- **Shocks:** the drought (fewer than 50 maker inserts in 0.5 s) gives a 3 s warning, with no false positives in this
  tape.
  - Pull every quote before onset + 3 s.
  - Sweep the stale side within 5–10 ms of the jump; read the direction from the first inserts at the new level.
  - Then quote into the burst.
- **Baselines on the full tape:** s_ladder −194k, sig_ladder −233k, robust_ladder +24.6k. The other session's
  `strategies/marek/r2_ladder.py` with the shock guard and volatile handling reaches +151k to +330k (its log is in
  `r2_ladder/results.jsonl`).
