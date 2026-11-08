# Round 3 findings: data and strategy

Written 2026-11-08. This page compiles what we know about round 3 (`data/round3.csv`) and what it means for our
round-2 strategies. Sources:
- **Tape statistics:** `data_explorer/round3/README.md`, and the side-by-side `data_explorer/compare-round2-round3/README.md`.
- **Generator reverse-engineering** (run in parallel by four agents, see `BRIEF.md` and the board `NOTES.md`):
  `flow/README.md`, `makers/README.md`, `novelty/README.md`. The `s_dynamics/` README is not written yet; its fit is
  `s_dynamics/params_v3.json` and the S path `s_dynamics/s_path.parquet`.
- **Strategy tests:** `strategy/` (this session). Harness `strategy/eval3.py`, raw rows in `strategy/results.jsonl`,
  summary `strategy/summ.py`.

**Bottom line.** Round 3 is round 2 plus a new "ladder" maker and transient impact. The round-2 approach (Bayesian S
filter, stale-quote sniping, shock sniper) still works, but **out of the box it loses 40 % of its edge**, because the
S filter mistakes ladder orders for ordinary makers. Two small fixes recover all of it and more:
`strategies/marek/round3/r3_lad.py` makes **RESULT_R3** on round 3 against 451k for `shockgen_best.py`, and
**RESULT_SYN** against 558k on six synthetic round-3 tapes. No change of approach is needed.

STRATEGY_SECTION

## A. What changed in the data (round 3 vs round 2)

### A1. The headline numbers

| | round 2 | round 3 | Comment |
| --- | ---: | ---: | --- |
| Events / s | 366 | 402 | ladder inserts (+≈ 8 /s calm, +≈ 42 /s volatile) |
| σ 1 s (ticks/√s) | 3.80 | 4.49 | more volatile time (28 % vs 15 %) |
| Variance ratio 2 s / 10 s / 60 s | 1.03 / 1.03 / 1.17 | 1.11 / 1.13 / **0.86** | moves now partly revert over 30–60 s (transient impact) |
| Touch size | 1,122 | 968 | more volatile time, where the touch is thin |
| Depth 2 / 4 / 7 ticks behind best | 292 / 37 / 9 | **752 / 536 / 178** | the ladder; drops to 14 at 8 ticks |
| Spread mean | 1.23 | 1.25 | 1 tick 82 % of the time |
| Taker markout 1 s | −0.17 | +0.10 | takers win short term, lose by 10 s (−0.09) |
| Price impact peak (after an aggressive order) | 0.56 at 1 s, flat after | 0.93 at 2 s → 0.58 at 20 s | transient impact |
| Price range | 9444–10017 | 9732–10160 | half of all impact washes out |

### A2. The generator: round 2 plus two mechanisms

Everything below is from the four agent READMEs; confidence is theirs.

1. **New ladder maker (certain).** Bursts of 1–6 orders on one side, exactly **2000 µs** apart, one tick further from the
   touch each, at **touch−2 … touch−7** of the same-side touch at insert time (start −2 in 61 %, else −3…−6; length 2:
   51 %, 3: 26 %, 4: 12 %, 5: 6 %, 6: 5 % among multi-rung bursts).
   - It anchors to the **touch, not S**. Sizes: one level B per burst, rungs B·U(0.75, 1.25); calm B ≈ LN(5.02, 0.34)
     (median 152), volatile LN(3.52, 0.31) (median 34).
   - Rate runs on the regime clock: ≈ 6.6 events/s in calm (5.5 of them single orders, invisible to the 2 ms test),
     ≈ 19.4 /s in volatile (8.2 singles). It rises after price activity and stops completely during shock droughts.
   - Probably 4 agents, each with at most one order per level; at most 24 live ladder orders per side.
   - Cancels: essentially never inside the band (0.02–0.3 /s), **immediately once the touch is more than 7 ticks away**
     (decision, then a 2 ms + Exp(52 /s) delay, mean 21 ms), and only ≈ 2.5–3 /s when the price comes to the order
     (base makers: α + 6·mv, ≈ 20 /s at mv 3).
   - Fills against ladder orders move S like any other fill. 6 % of calm and 24 % of volatile ladder orders fill.
2. **Transient impact (high confidence).** Only about half of each taker's impact on S is permanent. The rest decays:
   G(τ)/G(0.5 s) ≈ 0.40 at 120 s in calm and 0.52 in volatile. Fits: volatile c 0.062, permanent share 0.52–0.55,
   T 15–18 s; calm permanent share 0.45–0.58, T 17–45 s (the two agents disagree on calm). Shock jumps do **not** revert.
   The ξ noise reverts too. In round 2 (and round 1) impact was fully permanent.
3. **Parameter shifts (high confidence).** Volatile occupancy 28 % (r2 15 %): calm dwell 154 s (r2 247), volatile 60 s
   (r2 43). Calm impact c 0.0171, e 0.61 (r2 0.0142, 0.625). Volatile maker offset mean ≈ 1.5 (shape gamma-like). Jump sd
   47 (r2 43).
4. **Unchanged:** base makers (180 /s, round-1/2 size law and D law in calm, same lifetimes), takers (coin flips,
   lognormal(4.09, 1.40), 4/6 per side per s, limit round(S ± 1) / round(S ± 2.55) plus a tail), the shock schedule (27
   droughts, 3.0 s, U(≈145, 300) gaps, jump at drought end, taker burst), no informed flow, no Hawkes, no seasonality.

### A3. Small facts worth knowing

- **Book lags S more in volatile:** ACF of mid − S at 200 ms is 0.64 (r2 0.15). After an S jump > 0.5 in volatile the gap
  is 0.62 of its size at 200 ms and 0.27 at 500 ms (r2 ≈ 0 by 500 ms). About 30 % of the depth at −2…−7 is ladder, which
  ignores S.
- **Stale touches** (touch beyond S by > 0.5): calm 0.13–0.15 /s per side (r2 0.10), volatile 1.1 (r2 1.4); 97 % start
  within one order of a taker. Ladder orders are only 3–5 % (calm) / 17–21 % (volatile) of the stale touch quantity.
- **No ladder order is ever on the stale side at a shock jump** (ladders stop in droughts). The stale base quantity at
  jumps is 0–1428 lots per shock, the same as in r2.
- **Post-jump burst:** for 0.2 s after a jump ≈ 90 % of takers are on the jump side (≈ 4 extra), then a symmetric
  ≈ 22 /s per side decaying linearly to 4 /s by +3 s. First jump-side taker a median 77 ms after the jump.
- **Regime tells, fastest first:** exact-2 ms insert pairs (≈ 11 bursts/s volatile vs ≈ 1 calm) and the share of maker
  inserts at 20–70 lots flip within 1 s; maker quotes never within 0.7 of S; spread > 1.5 over 1 s.
- **Maker markouts on the tape** are −0.70 ticks/lot at a 1-tick spread and −1.7 in the high-volatility tercile (data
  explorer), i.e. passive quoting at the touch is still a losing game for the generator's makers; our edge comes from S.
- Not testable on a replayed tape: the ladder is touch-anchored, so in a live generator our own quotes would move it.

## C. Open questions and next steps

NEXT_SECTION
