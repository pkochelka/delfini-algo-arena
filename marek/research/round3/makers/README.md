# Round 3 makers: quoting, sizes and cancellation

Tape: `data/round3.csv`, 6000 s, 1,178,322 maker inserts (G) and 56,274 IOCs (F).
- **S:** I used s_dynamics' `s_path.parquet`. Most of the work used v0, the round-2 HMM with ladder orders left out of the
  likelihood.
  - On v1 (fitted transient impact) I re-ran the placement fits, the book and stale tables and the base cancel MLE.
    Only the vol D split changed.
  - Median posterior sd is 0.06 tick in calm and 0.10 in vol.
- **States** come from `states.py`:
  - **calm** and **vol** are flow's S-based segments (`flow/regimes.csv`, about 28 % vol).
  - **drought** runs from the shock onset to the jump. **post** is the 3 s after the jump (`flow/shock_events.csv`).
- **Ladder orders:** these are orders linked by an exact 2000 µs gap to an earlier insert (`chains.py` →
  `cache/chains.pkl`, with columns chain, rung k and length L). An order counts as "ladder" when 2 ≤ L ≤ 6.

## In short

Round 3's maker side is **round 2's maker process unchanged, plus one new participant, a "ladder" maker that runs on the
regime clock.**

| Part | Base makers (≈ 180 /s, both regimes) | Ladder maker (new) | Confidence |
| --- | --- | --- | --- |
| Who | round-1/2 makers | 4 independent agents (inferred), each holding at most 1 order per price level per side | high (4-cap) / medium (agent wake rule) |
| Rate | 180 /s, the same in calm and vol | calm ≈ 8–10 inserts/s, vol ≈ 42 /s (multi-rung bursts 1.1 vs 11.3 /s, see flow) | high |
| Size | round(LN(2.661, 1.800)), as in r1/r2 | rung = round(B·U(0.75, 1.25)), with B per burst: **calm LN(5.02, 0.34) (median 152), vol LN(3.52, 0.31) (median 34)** | high |
| Price | bid = min(⌊S − D⌋, ask − 1); calm D = 0.066 + Exp(0.45) (r1); vol D ≈ 0.57 + Exp(1.05) (mean 1.6, ≈ r2) | levels **touch−2 … touch−7** of the same-side touch at insert time, near→far, one insert every 2.000 ms; only the agent's missing levels | high |
| Cancel | decision hazard α + β·max(0, mv): α 1.7 (calm, vol), 5 (drought); **β ≈ 3.4 /tick** (r2 calm 6.1) | inside the band: ≈ 0; **beyond touch−7: immediate decision**; at touch / touch−1: ≈ 2.5–3 /s, weakly S-dependent | high / high / medium |
| Delay, decision → cancel row | **Exp(470 /s) + Exp(52 /s)**, mean 21 ms (exact event study on ladder far-edge cancels) | same | high |
| Droughts | as in r2 (α ×3–3.5, wider D) | stops entirely, so no ladder orders are stale at any jump | high |

The lead's "minutes with +15–20 % maker inserts and bigger median size" are the **volatile regime**. That regime is
switched on 28 % of the time in r3, against 15 % in r2. In vol the ladder adds about 34 /s of 20–70-lot orders. The calm-regime ladder
adds ≈ 8 /s of ≈ 150-lot orders that vanish in vol. Together these make the per-minute size and count swings.

## 1. Sizes (`sizes0-2.py`, `sizes_ts*.py`, `size_mix.py`, `ladder_size*.py`)

- **Per-minute and 10 s windows.**
  - Wide-spread windows have +30–40 /s extra orders at 20–70 lots, peaking around 35 lots.
  - 1-tick-spread windows instead have +6–8 /s extra at 90–320 lots, peaking around 150 lots. That excess disappears in the
    wide windows.
  - corr(20–70 excess, spread) = 0.66 over 600 windows.
  - The 1–10-lot rate is the same everywhere (≈ 78 /s).
- **Mixture MLE on non-chain makers** (rounded lognormal base + lognormal ladder singletons):

| Regime | base μ | base σ | ladder-single weight | ladder μ, σ |
| --- | --- | --- | --- | --- |
| calm | 2.661 | 1.800 | 2.9 % | 5.006, 0.317 |
| vol | 2.662 | 1.806 | 4.5 % | 3.484, 0.351 |
| round 2 (all) | 2.657 | 1.803 | — | — |

  So the base size law is exactly round 1/2's. flow confirms this independently: non-ladder counts per size band match
  180·P(band) to ±0.3 /s outside the ladder bands.
- **Within a burst** the rung sizes are B·U(0.75, 1.25), rounded. The max/min ratio quantiles (50/90/99 %) are:

  | Burst length | observed | simulated |
  | --- | --- | --- |
  | L = 4 | 1.36 / 1.56 / 1.64 | 1.37 / 1.55 / 1.62 |
  | L = 6 | 1.45 / 1.59 / 1.65 | 1.46 / 1.58 / 1.64 |

  U(0.8, 1.2) and a lognormal (sd 0.115) both fit worse. Size does not depend on rung index or side.
- **The burst level B** is independent from burst to burst (lag-1 corr 0.09):

  | Regime | log B | B quantiles |
  | --- | --- | --- |
  | calm | 5.02 ± 0.34 | 5–95 %: 74–276 |
  | vol | 3.52 ± 0.31 | 5–95 %: 19–58 |

  B switches within the same 1 s bin as the regime (flow, `switch_align.py`). The calm/vol ratio is 4.5.
- No round-number preference: ladder and base prices are uniform mod 10.

## 2. Ladder placement (`chains.py`, `ladder_occ.py`, `stack.py`, `agents_test.py`, `burst_*.py`, `sweep.py`, `wake*.py`)

- **Rung clock.** 45k pairs of inserts sit exactly 2000 µs apart (background ≈ 240 at ±7 µs). 99.8 % are on the same
  side, and the price steps one tick away from the touch (95.8 %). Pairs at 4000 µs exist only inside chains. So the
  clock never skips a slot: a burst is consecutive inserts.
- **Band.** 99 % of bursts lie inside [touch−7, touch−2] of the same-side touch *at insert time*.
  - The fit is best at lag 0. Using the touch from 10 ms earlier drops it to 93 %.
  - Against S the placement is looser: in calm, S − p0 has an IQR of 2.35–2.99 for bursts starting at touch−2. So the
    reference is the touch, not S.
  - Start-level and length shares:

    | Start level | −2 | −3 | −4 | −5 | −6 |
    | --- | --- | --- | --- | --- | --- |
    | share | 61 % | 10 % | 9 % | 10 % | 10 % |

    | Length L | 2 | 3 | 4 | 5 | 6 |
    | --- | --- | --- | --- | --- | --- |
    | share | 51 % | 26 % | 12 % | 6 % | 5 % |

    Length-1 bursts are invisible to the 2 ms test.
  - No burst extends past −7.
- **Four agents.** At most 4 live ladder orders share one price level:
  - In calm, occupied levels hold 1/2/3/4 orders in 3625/2119/1126/403 cases. A 5th appears only in 9 of 37k cases, which I
    read as a pending cancel.
  - At burst start, the levels chosen hold 0–3 live ladder orders (4 in 22 of 34k cases). Band levels outside the burst
    hold 4 in 12 % of cases.
  - So each burst is one agent filling **its own** missing levels near→far. A level the agent already holds is skipped
    without using a 2 ms slot, which gives the 2–5-tick steps (4 % of steps).
  - Each side holds at most 24 live ladder orders. flow's "ladder hazard falls with live ladder count" is this capacity.
- **When an agent acts:** I only know this roughly. flow has the timing: no clock, and the rate rises with recent
  |ΔS| or touch changes.
  - Near-end bursts (start −2) follow the touch moving away. For L = 2 the touch moved by about +2 in the previous 200 ms.
  - Far-end bursts (end −7) follow the touch moving toward the ladder.
  - After a calm 1-tick move toward the ladder that then holds for 1 s, the new −7 level gets 1–3 ladder inserts:

    | inserts | 0 | 1 | 2 | 3 | 4 |
    | --- | --- | --- | --- | --- | --- |
    | share | 2 % | 52 % | 37 % | 10 % | 0.1 % |

    The first comes after a median of 100 ms. That is fewer than 4, so not every agent refills every gap within a second.
    How an agent decides to wake is still open.
- **Ladder depth** (`depth_by_level.csv`, chain orders only, per side): in calm, 175–245 lots at each of levels −2…−6.
  Calm ladder-like singletons (70–330 lots, ≥ 2 behind the touch, 7 /s) cancel exactly like chain ladders (table in §3).
  Counting them, the ladder is most of the depth at levels −3…−7 in calm. In vol, chains alone are 16–35 % of the depth at −2…−7.

## 3. Cancellation

### Ladder (`ladder_hazard.py`, `ladder_cancel.py`, `singletons.py`, `ladder_delay.py`)

**Hazard of a live ladder order by distance from the same-side touch.** The touch is taken at the cancel row.

| rel to touch | ≤ −8 | −7 | −6 | −5 | −4 | −3 | −2 | −1 | 0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| calm chain | 44 | 0.33 | 0.19 | 0.15 | 0.02 | 0.02 | 0.03 | 2.5 | 4.4 |
| calm ladder-like single | 42 | 0.29 | 0.21 | 0.23 | 0.03 | 0.06 | 0.22 | 2.7 | 5.2 |
| calm base single at the same levels | 2–6 | 0.7 | 1.4 | 1.0 | 1.0 | 1.5 | 1.9 | 3.1 | 6.2 |
| vol chain | 44 | 2.9 | 1.1 | 0.8 | 0.08 | 0.06 | 0.08 | 3.6 | 6.5 |

- **Far edge.** An order beyond touch−7 is cancelled at once, subject to the delay. The residual hazard at −5…−7 comes
  from the touch flickering to −8 and back inside the delay.
- **Near zone (touch, touch−1).** With the state taken 25 ms before the row, the hazard is about 2.5–3 /s.
  - It rises only weakly as S passes the order: 1.3 → 4.1 /s in calm as S − p goes from > 3 to ≈ 0.
  - It is almost flat in mv: 1.6–3.7 /s.
  - So there is no β·mv term like the base makers have. This is the least certain part of the ladder law.
- **Lifetimes** (insert to cancel row) and fill shares:

  | Ladder state | p10 (s) | p50 (s) | p90 (s) | p99 (s) | filled |
  | --- | --- | --- | --- | --- | --- |
  | calm | 0.45 | 2.8 | 10.8 | 25.9 | 6 % (partial fills included) |
  | vol | 0.14 | 0.60 | 1.7 | 3.6 | 24 % |

  These lives make up the r3 lifetime tail.
- Fills against ladder rungs move S like any other fill (novelty).

**Exact delay law.** I timed 15.9k ladder orders from the touch change that put them beyond −7 (an exact timestamp) to
their cancel row:

| ms | 0–1 | 1–2 | 2–3 | 3–4 | 4–5 | 5–10 | 10–20 | 20–30 | 30–40 | 40–60 | 60–100 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| density /s | 11 | 24 | 33 | 37 | 41 | 38–35 | 32–22 | 19–13 | 10–8 | 6–3 | 1.7–0.6 |

- MLE fits:
  - **Exp(470) + Exp(52)**, mean 21.3 ms: nll −45760.
  - Gamma(k 1.37), mean 21.3 ms: nll −45663.
  - L0 + Exp: L0 → 0, nll −45235.
- So the "≈ 5 ms dead time + Exp(≈ 50)" of round 2 is a short ≈ 2 ms exponential stage followed by Exp(≈ 50 /s).
- The mean, 21 ms, matches round 1's Exp(40) and round 2's 23–25 ms.
- I assume the base makers share this delay. Their event study in round 2 (S jumps at takers) gave the same rate and
  plateau.

### Base makers (`cancel_mle.py` → `cancel_mle.json`, `cache/cancel_mle_30k.log`; `lifetimes.py` → `lifetimes.csv`)

**Direct MLE.** The method is round 2's: decision hazard α + β·max(0, mv) per state at insert, then an Exp(b) delay to
the row. Fills are included, because the row comes anyway. The sample is 15k random base orders on a 2.5 ms grid.

| S | α calm | β calm | α vol | β vol | α drought | β drought | α post | β post | b (/s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v1 (`cancel_mle.json`) | **1.69** | **3.44** | 1.73 | 3.39 | 5.07 | 3.2 | 1.93 | 3.1 | 64 |
| v0 (`cache/cancel_mle_v0.json`) | 1.71 | 3.44 | 1.74 | 3.37 | 4.94 | 3.2 | 2.08 | 2.1 | 63 |
| round 2 | 1.73 | 6.10 | 1.92 | 4.47 | 6.07 | 4.3 | 1.83 | 6.9 | 45 |

**Model-free hazard by mv** (calm / vol, age > 50 ms, v1):

| mv | ≤ −0.5 | −0.5–0 | 0–0.1 | 0.1–0.25 | 0.25–0.5 | 0.5–0.75 | 0.75–1 | 1–1.5 | 1.5–2 | 2–3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| calm | 1.64 | 1.66 | 1.83 | 2.44 | 3.30 | 4.31 | 4.81 | 5.70 | 5.52 | 4.47 |
| vol | 1.69 | 1.63 | 1.96 | 2.99 | 3.21 | 4.00 | 5.83 | 7.06 | 8.57 | 10.0 |

- **α is unchanged:** 1.7 /s in calm and vol, and ×3 in droughts. Favourable moves still don't matter.
- **β in calm is lower than round 2's, ≈ 3.4 vs 6.1 per tick.** Calm hazard reaches only 5–6 /s at mv 1–2 (r2: 8–13).
  - It is the same on S v0 and v1.
  - The ≈ 4 % hidden ladder singletons in the calm sample (hazard ≈ 0) cannot explain a halving.
  - It is either a real change, or a consequence of transient impact: makers may react to the permanent part of S
    rather than the full kick.
  - Medium confidence. I'd check it on a synthetic tape with r2's β and the r3 impact law.
- In vol, β ≈ 3.4 matches round 2's 3.4–4.5.

**Lifetimes are round 2's to within 0.02 s:**

| Base state | mean (s) | p10 | p50 | p90 | filled |
| --- | --- | --- | --- | --- | --- |
| calm | 0.457 | 0.078 | 0.326 | 0.987 | 14.4 % |
| vol | 0.379 | 0.066 | 0.245 | 0.853 | 16.2 % |
| drought | 0.213 | 0.045 | 0.154 | 0.439 | 6.1 % |

Every order gets a cancel row, filled or not. Lifetime does not depend on size: 0.43–0.47 s across size bands.

## 4. Base placement (`offset.py`, `offset_vol.py` → `offset_by_state.csv`, `offset_splits.csv`)

This is the censored MLE under the round-1 clip rule, with S integrated over its posterior. It adds a mixture component
of weight w at uniform touch−2…−7, which absorbs the hidden ladder singletons. Chain ladder orders are excluded.

The table uses the final S, v1 (`offset_by_state.csv`, log in `cache/offset_v1.log`). On S v0 the calm row was the same
(0.065 + Exp(0.462)), and vol came out at 0.11 + Exp(1.47).

| State | n | d0 | scale | mean D | w ladder | share at the clip | one-multiplier m | Δnll/obs (one-mult − free) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| calm | 777k | 0.066 | 0.450 | 0.516 | 2.5 % | 0.50 | 1 | — |
| calm, S sd < 0.06 | 100k | 0.064 | 0.443 | 0.507 | 2.4 % | 0.51 | | |
| vol | 313k | 0.574 | 1.050 | 1.62 | 2.2 % | 0.19 | **2.66** (r2 2.62) | +0.090 |
| drought | 4.4k | 0.43 | 2.18 | 2.61 | 0 | 0.02 | 5.0 | +0.03 |
| post | 15k | 0.052 | 0.611 | 0.66 | 2.2 % | 0.41 | 1.28 | +0.05 |

- **Calm is the round-1 law**, 0.05 + Exp(0.45). It is the same by side (scale 0.462 / 0.458), by 1000 s segment
  (0.453–0.470) and by size band (0.44–0.45). The only exception is the 70–320-lot band, where the ladder weight is 14 %
  and the fit is slightly off.
- **The vol shape is not a plain exponential from ≈ 0.**
  - On S v1, the shifted exponential 0.57 + Exp(1.05), mean 1.62, wins over a start near 0. This agrees with
    s_dynamics' fitted 0.67 + Exp(1.00).
  - Nelder–Mead also finds a worse local optimum, 0.175 + Exp(1.46) (`offset_vol.py`). From that start, a Gamma does
    better: 0.17 + Gamma(k 3.1, θ 0.45), mean 1.56. That is the same story: a mode at ≈ 1 tick, not at 0.
  - r2 vol was 0.70 + Exp(0.90), mean 1.6, so the vol law is about unchanged (maybe a slightly longer tail).
  - Mean D_vol / D_calm ≈ 3.1. The one-multiplier model m·D_calm (m = 2.66) is clearly worse than the shifted fit.
  - Medium confidence on the exact shape.
- No maker insert crosses. Clipping is as in rounds 1 and 2.

## 5. Book state (`book_stats.py` → `book_by_state.csv`, `depth_by_level.csv`; `book_vs_s.py` → `stale_episodes.csv`, `stale_ladder.csv`)

| State | time (s) | sd(mid − S) | spread 1 / 2 / 3+ | mean spread | touch qty | E[S − bid], E[ask − S] |
| --- | --- | --- | --- | --- | --- | --- |
| calm | 4177 | 0.54 (r2 0.44) | 94.4 / 5.5 / 0.1 % | 1.06 | 1208 | 0.52, 0.54 |
| vol | 1659 | 1.71 (r2 1.50) | 53.6 / 35.6 / 10.8 % | 1.59 | 417 | 0.75, 0.84 |
| drought | 81 | 1.17 | 6 / 5 / 90 % | 3.82 | 224 | 1.89, 1.94 |
| post | 80 | 3.33 | 73 / 18 / 9 % | 1.62 | 655 | 0.79, 0.84 |

- **Gap persistence.** ACF of mid − S on a 10 ms grid:
  - calm: 0.88 at 50 ms, 0.77 at 100 ms, **0.58 at 200 ms** (r2 0.50), 0.23 at 500 ms;
  - vol: 0.88 at 50 ms, 0.78 at 100 ms, **0.64 at 200 ms** (r2 0.15), 0.39 at 500 ms.
- **After an S jump of more than 0.5 in vol,** the gap goes 1.20 → 0.86 at 100 ms → 0.62 at 200 ms → 0.27 at 500 ms.
  r2 was ≈ 0 by 500 ms. The vol book catches up with S more slowly than in r2.
  - Part of this may be S v0 artefact: the transient impact is not in the HMM prior, so S may overshoot.
  - Part is real: about 30 % of the depth at −2…−7 is ladder, which ignores S.
- **Stale touches** (touch beyond S by more than the threshold), starts per second per side:

  | thr | calm (r2) | vol (r2) | median duration | > 5 ms |
  | --- | --- | --- | --- | --- |
  | 0.5 | 0.13–0.15 (0.10) | 1.09–1.15 (1.34–1.44) | 190 ms calm / 107 ms vol | 96–99 % |
  | 1.0 | 0.03 | 0.73–0.79 | | |

  - Edge at the start: 0.7 calm, 1.3 vol (0.5 thr). Calm episodes are longer than r2's (150–160 ms).
  - 97 % start within one order of a taker (r2 60 %, a different S).
- **Ladder share of the stale touch quantity:** 3–5 % in calm, 17–21 % in vol. Ladder-dominated stale touches last about
  as long as the others (88–107 ms vs 106–109 ms in vol). The ladder does not create a separate class of stale quotes.
- **Shocks.** At none of the 27 jumps did a ladder order rest on the stale side (`shock_ladder.py`). The stale base quantity
  is 0–1428 lots per shock, the same picture as r2.

## Validation
- Base sizes, rate, lifetimes, calm D and calm book reproduce round 2 to 2–3 digits.
- The ladder size law, the U(0.75, 1.25) rung noise and the 4-per-level cap were each checked against a simulation or a
  null:
  - the 4000 µs pairs against a background at a µs offset;
  - chosen levels against levels outside the burst.
- The ladder cancel band is the same in calm and vol, and the same for chain orders and for calm ladder-like singletons.
- **Not done:** a full simulation of the 4-agent ladder (band + wake rule + cancel law) compared against the tape's burst
  length, start-level and depth distributions. The wake rule is not pinned down enough for that yet.

## Still open
1. **Agent wake rule.**
   - Agents are not periodic and not a fixed latency after a touch change.
   - Their rate rises with recent price movement and falls with the agent's fill level (flow).
   - Fewer than 4 agents refill a gap within 1 s.
   - Possibly each agent wakes on its own Poisson clock, at a rate that rises with recent |ΔS|.
2. **Near-zone ladder cancel.** It is about 2.5–3 /s with weak S dependence. Is it the generic α of a base order with
   β ≈ 0, or a separate rule?
3. **Vol D shape.** Gamma k ≈ 2.4–2.9 or a shifted exponential? Refit on the final S. The transient-impact model may change S
   in vol.
4. **Calm β ≈ 3.4 vs r2's 6.1.** Is it real, or an S/transient-impact artefact? The MLE sample also holds ≈ 4 % hidden
   ladder singletons in calm.

## Implications for a market maker / taker
- **Calm (72 %): the round-1/2 playbook still holds for the base book.**
  - The book is S ± (0.05 + Exp(0.45)), clipped at the touch.
  - Base orders die at about 1.7 + 3.4·mv /s, after a delay of 2 ms + Exp(52) (mean 21 ms).
  - Requote within 5–10 ms of an S jump.
- **The ladder is free, slow depth at touch−2…−7.**
  - Up to 4 × 150 lots per level in calm and 4 × 34 in vol.
  - It is never cancelled while it sits inside the band, and pulled only once the touch is more than 7 away.
  - It does not react to S. When S runs *toward* a ladder side, the rungs end up at the touch and are cancelled at only
    about 2.5–3 /s. Base orders would be cancelled at α + 6·mv, which is ≈ 20 /s at mv 3.
  - So after a fast multi-tick move, the touch on the side the price moved toward can be a ladder order that is still
    stale relative to S. Sweeping it is profitable when S − price exceeds the impact we cause.
  - In practice, ladder-dominated stale touches are only 4 % (calm) and 16 % (vol) of stale episodes. The edge is a small
    add-on, not a new strategy.
- **Queue position.** A ladder order joining a level never leaves while it stays in the band. So a deep level fills up with
  sticky ladder size, and our own queue position behind it is worse than the round-2 depth suggests. Join levels −2…−7
  before the ladder does, or avoid them.
- **Regime detection.** The share of maker inserts at 20–70 lots, or exact 2000 µs insert pairs (≈ 11 bursts/s in vol vs 1
  in calm), flips within 1 s of a regime switch. That is faster and cleaner than spread-based detection.
- **Shocks: same as r2.** Pull at drought onset + 2.9 s and sweep the stale base side within 5–10 ms of the jump. Ladders
  stop in droughts, so they add nothing to the stale book.

## Files
| File | What it does |
| --- | --- |
| `ladder_model.csv` | the ladder participant's parameters in one table |
| `prep.py` | order table + quotes + trades → `cache/` |
| `chains.py` | exact-2000 µs ladder chains → `cache/chains.pkl` |
| `s_load.py`, `states.py` | S v0 per insert, ladder flag, calm / vol / drought / post |
| `regime_sz.py` | my size-band Poisson HMM regime (98.9 % agreement with S-based labels) |
| `sizes0-2.py`, `sizes_ts*.py`, `size_mix.py` | size laws, episode comparison, mixture MLE |
| `ladder_size*.py` | rung noise U(0.75, 1.25), B per regime |
| `ladder_occ.py`, `stack.py`, `agents_test.py`, `sweep.py` | band, stacking, 4-agent cap, slot clock |
| `burst_*.py`, `wake*.py`, `ladder_vs_s.py` | what precedes a burst, refill counts, touch vs S anchor |
| `ladder_hazard*.py`, `ladder_cancel.py`, `ladder_near.py`, `singletons.py`, `ladder_delay.py` | ladder cancel law, exact delay |
| `offset.py`, `offset_vol.py` | base D per state with the ladder mixture |
| `cancel_mle.py` | base cancel hazard MLE and hazard tables |
| `lifetimes.py`, `book_stats.py`, `book_vs_s.py`, `shock_ladder.py` | lifetimes, book by state, depth by level, stale episodes, shocks |
