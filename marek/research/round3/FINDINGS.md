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
`strategies/marek/round3/r3_lad.py` makes **795k** on round 3 against 451k for `shockgen_best.py`, and
**923k** on six synthetic round-3 tapes (shockgen_best's filter without the ladder fix: 558k), and 862k on four held-out synthetic tapes (shockgen_best: 356k). No change of approach is needed. The brainstorm and development round (section D) raised this to **853.7k** (`r3_dev_best.py`).

## B. Strategy findings

### B1. Round-2 strategies on the round-3 tape, unchanged

PnL net of the breach penalty, split by period (`strategy/eval3.py`). Shock = onset −1 s … +20 s; regimes from
`flow/regimes_1s.csv`.

| Strategy | Round 3 | calm | volatile | shock | Round 2 (same split) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `strategies/peter/fvs_mm.py` | 485.0k | 117.0k | 290.4k | 77.5k | 606.8k |
| `strategies/marek/round2/final3.py` | 457.0k | 120.1k | 229.9k | 107.1k | |
| `strategies/marek/round2/overfit_best.py` | 455.4k | 118.6k | 227.8k | 109.0k | |
| `strategies/marek/round2/shockgen_best.py` (r2 best) | 451.0k | 118.6k | 227.3k | 105.1k | 772.4k (calm 305.6k, vol 211.3k, shock 255.5k) |
| `strategies/marek/round2/r2_ladder.py` | 424.3k | 87.5k | 316.1k | 20.8k | 365.3k |
| `strategies/marek/round2/snipe_best.py` | 79.5k | −92.3k | 157.3k | 14.5k | |
| `strategies/marek/round1/robust_ladder.py` | −11.2k | 12.0k | −14.8k | −8.5k | |
| `strategies/marek/round1/s_ladder.py` | −493.0k | 53.9k | −447.8k | −99.1k | −163.9k |

- Everything that does not track S with the round-2 model loses money, as in round 2.
- The S-filter family stays positive, but **calm PnL per second halves** (≈ 31 /s vs ≈ 66 /s on r2) and volatile per
  second drops by 40 %, although the base book is the same. That pointed at the filter, not at the market.

### B2. Fix 1: filter out the ladder maker (+280k, the main result)

The FV filter treats every market insert as an S-maker: bid = ⌊S − D⌋ with D = 0.05 + Exp(0.45) in calm. Ladder rungs
sit 2–7 ticks behind the touch regardless of S, so under that law they look like strong evidence that S is low (for
bids) and, through their width, that the regime is volatile. In volatile they are ≈ 14 % of all maker inserts.

`strategies/marek/round3/r3_lad.py` (flag `lad_on`):
- Drop any insert that arrives exactly 2000 ± 3 µs after the previous market insert on the same side, one tick further
  out (rung 2+).
- Hold any insert that sits ≥ 2 ticks behind its same-side touch for 2.01 ms; drop it if a rung follows, else feed it to
  the filter then. Rung 1 is caught this way.
- Do not hold inserts during a detected drought (`lad_dr`): post-jump inserts are urgent and ladders stop in droughts
  anyway. Holding them cost 19k of shock PnL.
- `lad_ss`: ignore possible rungs in the shock sniper's "far insert" direction rule. Neutral (−3k), kept on.

| Variant (round 3) | PnL | calm | volatile | shock |
| --- | ---: | ---: | ---: | ---: |
| filter off (= shockgen_best) | 451.0k | 118.6k | 227.3k | 105.1k |
| rung 2+ dropped only, no holding | 724.2k | 166.7k | 419.7k | 137.8k |
| rung 2+ dropped + deep inserts held (also in droughts) | 731.1k | 191.4k | 419.7k | 120.0k |
| + no holding in droughts (`lad_dr 1`) | 749.7k | 190.7k | 419.5k | 139.5k |

Synth (6 seeds, with round-3 constants): filter off 557.5k → on 841.8k (+284k).

### B3. Fix 2: a larger outlier share in the filter (+40k)

About 3 % (calm) to 4 % (volatile) of maker inserts are single-order ladder placements that the 2 ms test cannot see.
The filter's uniform outlier share eta (0.005 in round 2) is too small to absorb them.

| eta | Round 3 | Synth mean (6) |
| --- | ---: | ---: |
| 0.005 | 757.6k | 841.8k |
| 0.05 | 795.6k | 882.8k |
| 0.08 | 771.9k | 883.8k |
| 0.10 | 778.1k | 897.2k |
| 0.12 | 798.3k | 901.9k |
| **0.15** | **796.4k** | **907.9k** |
| 0.20 | 778.3k | — |

Calm PnL rises monotonically up to 0.12–0.15; the round-3 total wobbles with the shock part (see B5).

### B4. Smaller tuning (round 3 / synth-mean deltas, base = eta 0.1)

| Change | Round 3 | Synth | Kept |
| --- | ---: | ---: | --- |
| s_dynamics v3 calm c 0.0171, e 0.61, D scale 0.4756, volatile D scale 1.06 (vs r2 constants) | +8k | +116k* | yes |
| transient impact in the filter (`tr_on`; S drifts back by the decaying share, T 17.3 / 15 s; reset at a drought, `tr_sr`) | +2.0k | +4.2k | yes |
| calm theta 0.25 (0.30) | +4.2k | +6.4k | yes |
| volatile take_t 0.3 (0.15) | +2.0k | +4.2k | yes |
| pv_thr 0.7 (0.5) | +0.8k | +6.6k | yes |
| shock trigger on a taker sweeping ≥ 2 levels (`ss_trade`) | 0 | +2.7k | yes |
| eta 0.15 | +18.3k | +10.7k | yes |
| calm theta 0.35 | +14.3k | −0.7k | no |
| take_sd_t 1.0 / take_t 0.05 | +12.8k / +13.3k | −16.5k / −6.4k | no (shock luck on r3) |
| pv_thr 0.3 | +1.9k | −2.3k | no |
| ss_lim 10 (8) | −4.8k | −0.7k | no |
| xs_cap 600 / 1200 (950) | −0.9k / −1.8k | +0.7k / **−68.6k** | no |
| volatile passive quotes (theta_t 2.0 / 2.5) | −38.6k / −13.4k | −12.9k / −1.7k | no |
| theta_t, theta_keep_t, reach_t alone | 0 | 0 | no effect: passive quoting is off in volatile |
| shock sniper off (`ss_on 0`) | −16.1k | −46.8k | (the sniper is worth ≈ 47k) |
| skew quotes by the transient stock, y_k 0.05 / 0.1 / 0.2 / 0.5 / 1.0 | −5k / −53k / −222k / −456k / −590k | — | **no**: it also moves IOC take prices |
| the ladder filter off, at eta 0.1 | — | −131.6k | (the filter is still worth that much after the eta fix) |

\* vs the synth tapes generated with these same constants, so the synth number overstates it; the round-3 +8k is the
honest figure.

### B5. Final file and validation

`strategies/marek/round3/r3_best.py` = `r3_lad.py` + the "kept" rows of B4.

| Tape | shockgen_best (r2 best) | r3_lad (eta 0.1) | **r3_best** |
| --- | ---: | ---: | ---: |
| round 3 | 451.0k | 778.1k | **795.0k** (calm 238.6k, vol 421.7k, shock 134.7k, breach 17.7k) |
| synth v1 (6 seeds, used for selection) | — | 897.2k | **923.1k** |
| synth v2 (4 seeds, held out: makers' ladder law, stronger decay) | 356.0k | 852.7k | **861.7k** |
| round 2 (regression) | 772.4k | 772.6k | 761.0k (−11k, shock windows) |

- **Shock PnL is heavy-tailed and fragile on round 3.** One shock (4127 s, J = +124) flips between +0.2k and +17.5k
  on small parameter changes. There the first sign of the jump is a 402-lot taker sweeping the stale asks at 4129.835,
  ≈ 50 ms before any maker quotes at the new level, so the maker-insert trigger is sometimes too late. Judge shock
  changes on synth.
- Per period r3_best now earns at round-2 rates: calm ≈ 63 /s (r2 66), volatile ≈ 254 /s (r2 264), shocks ≈ 5k each
  (r2 ≈ 6k excluding the 5398 outlier).
- `peter/fvs_mm.py` has the same filter and should gain from the same fix, but our lineage is ahead of it on both tapes.

### B6. Tools added

| Path | What |
| --- | --- |
| `strategy/eval3.py` | runs files on round3 / synth / round2, splits PnL into calm / volatile / shock and per shock, 12 × 500 s windows |
| `strategy/ab.sh`, `strategy/batch*.sh`, `strategy/summ.py` | parallel A/B configs; summary table with deltas vs a base tag |
| `strategy/r3_synth.py` | round-3 synthetic generator: r2_synth + ladder maker + transient impact + r3 regime rates and burst head |
| `data/synth/r3_v1_seed1-6.csv`, `r3_v2_seed11-14.csv` | synth tapes (v2: makers' ladder law U(0.75, 1.25) rungs, calm hazard 0.25 at −5…−7, decay calm 0.45/30 s, vol 0.52/18 s) |
| `data_explorer/r3_v1_seed1/`, `compare-round3-r3_v1_seed1/` | synth vs real fingerprint |

**Synth fidelity:** events/s, touch size, spread, insert/taker sizes and the depth bump at 2–4 ticks match the real
tape; σ is ≈ 18 % too high (5.3 vs 4.5 at 1 s), the ladder is thinner at 5–7 ticks in v1, and impact decays less than
on the tape (variance ratio 60 s: 0.89 vs 0.86). Good for A/B of mechanisms, not for absolute levels.


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

## D. Brainstorm and development (five agents + one dev agent)

Reports: `ideas/{ceiling,bigmoves,volatile,calm,novel,develop}.md`. After the brainstorm the user dropped synth testing
(slow, not informative), so the final stack is accepted on round 3 only.

**Best file: `strategies/marek/round3/r3_dev_best.py`: round 3 853.7k (+58.7k over r3_best), round 2 778.4k (+17.4k).**
Verified with eval3: calm 254.5k, volatile 430.1k, shock 169.0k, breach 17.1k. It leads r3_best in all 12 windows of
500 s. Same imports as r3_best, no `modify_order`.

| Kept change | Round 3 alone | In stack (leave-one-out) |
| --- | ---: | ---: |
| bigmoves C2: early shock trigger on a ≥ 3-tick one-sided sweep from onset + 2.90 s, clip edge credit (`bm_sw*`, `ss_clip_e 2`) | +34.0k (shock; +28k of it is the 4127 s shock) | +31.6k |
| calm c2: take 0.10, reduce-only calm IOC at edge ≥ 0 + 1·sd, theta_keep 0 (also standalone `r3_calm_c2.py`, 811.5k) | +16.5k | +10.0k |
| volatile vfm20: one flatten-only passive quote at F ± 2 once the book-lag carry has run out | +12.1k | +8.5k |
| final s_dynamics filter constants | +7.3k | +5.4k |

**Why nothing doubles (all agents agree):**
- **The ±100 limit binds.** |pos| ≥ 95 for 59 % of calm and 90 % of volatile time. 73 % of volatile stale-quote events
  find no room (358k of flat-start value). Volatile cap 200 would gross +425k, but holding it costs ≈ 1,560 s of breach.
- **The filter and latency are not the limit.** An exact-S oracle adds +34k. Regime detection lags the true switch by
  ≈ 40 ms. 97–98 % of stale lots are still there after 5 ms.
- **Volatile momentum (new).** When the book lags S, the next 1–3 s S move ≈ 0.6 × (S − mid). Snipe edge is 1.31/lot at
  20 ms, 1.70 at 1 s and 2.13 at 5 s, and the carry is ≈ 40 % of volatile PnL. Every exit loses: IOC unwind −133k, IOC exit
  after 1–3 s −1.5k to −34k, ungated passive exits −4k to −17k.
- **Oversize outside shocks is dead.** Exit depth within 1–4 ticks of S after a big move is a median 0 lots. The ladder sits
  3–8 ticks below our fill, so selling into it loses.
- **Shocks.** Ceiling ≈ 104k (with oversize) against 58k captured before C2; the cap and IOC limit never bind. Round 2's
  higher shock PnL was one shock (5398 s, 251k stale value).
- **Calm.** Adverse selection from the filling taker's impact costs 0.56/lot (≈ 337k). Queue-first passive ceiling ≈ 365k
  against 123k realised, but every way of buying priority with room lost (pre-join −62k, deeper split −54k to −126k,
  over-cap seniority −46k plus penalty).
- **Simulator.** No exploitable rule: the breach penalty is an exact time integral, and the book never crosses, so an instant
  unwind always pays the spread.
- **Fading transient impact** loses everywhere (calm takes −24k to −193k, passive ±−90k, volatile −42k to −236k). The hold
  is 1–2.5 s; the decay takes 15–40 s.

**Bug to know about:** with `modify_order`, a REJECTED reply to a failed modify makes the base handler drop a live order,
and the position runs past the cap (one test scored −20M). `ideas/calm_a.py` counts pending modifies per order to fix it.

**Remaining leads:** a single volatile exit rule at the takers' limit round(F ± 2.55) with the momentum gate; calm queue
priority without spending room; ≈ 45k of shock value left (shocks 2, 14, 20).

## C. Open questions and next steps

1. **The 2× question:** see section D. The gap is volatile position room; a cheap exit is the open problem.
2. **Shock trigger latency** (4127 s case): fire on the first trade far through the stale touch, not only on maker
   inserts at the new level.
3. **Transient impact as an inventory signal** (not as a quote skew): untested.
4. **Port the ladder filter to `peter/fvs_mm.py`** if that lineage is still used.
5. **Synth fidelity:** fit σ (impact c or decay) and the ladder wake rule; regenerate.
6. **Round 4 robustness:** the ladder filter keys on an exact 2000 µs gap and the touch−2…−7 band. If round 4 changes
   the ladder's clock, those constants must be re-measured (the eta increase is the generic fallback).
7. **Commit:** none of this is committed yet (strategy files, `marek/research/round3/strategy/`, this file).


## E. HMM regime angle (2026-11-08)

Details: `hmm/README.md`. A fitted multi-stream Poisson HMM recovers the generator's three regimes (calm, volatile,
3 s drought; BIC prefers K = 3 among real regimes, K ≥ 4 only splits bursty ladder activity), all already modelled.
Quotes at the takers' own limit price capture impact-free residuals (+2.6/lot at 1 s in volatile) but only on the
momentum side where room is already used; every way of making room for them lost (−9k to −90k). Hindsight DP: volatile
take selection optimal 345k vs greedy 240k, no threshold rule beats greedy. Sweep of 63 knobs on r3_dev_best: flat.
New file `strategies/marek/round3/r3_hmm.py` (take_sd 1.0, take_sd_t 1.0): **857.6k** (+3.9k, noise level).

## F. Push sprint (2026-11-08 05:10-05:45)

Six agents on `r3_hmm.py` (857.6k), real tape only, tags xs*/xr*/xf*/evo* in `strategy/results.jsonl`.
- **Kept:** `dr_theta 1.0, dr_margin 0.2` → `strategies/marek/round3/r3_push.py` **863.3k** (calm −0.8k, vol +0.1k, shock +6.4k;
  +9.7k without the biggest shock). Fragile: dr_theta 1.1 or dr_margin 0.15 lose 7-10k (shock #18 flips −12..−15k).
- **Dead:** size_top 15-75 (−13..−21k), min_post, take_q (−9..−35k), cap_t 90 (−48k), volatile reduce-only IOC
  (`r3_x_room.py` tr_take_t, flat/−17k), extra take edge past |pos| 70 (rt_*), base-maker-cancel F shift
  (`r3_x_filt.py` ca_*, −9..−85k both signs), taker-stop S bound (tk_*, calm negative everywhere), lad_wait/lad_tol, grid
  h 0.025, d0/d0_v, max_sd, ss_shots/gap/hold, sg_thr/sg_win, shock_lead/len, shock_flat, ps_on, dr_fu 2.
- **Evolutionary tournaments** (3 agents, champion vs 2 mutants per 500 s window tape `data/r3win/round3_wNN.csv`, ~10 s/run):
  every evolved champion lost on the full tape (calm −1.3..−10.2k, volatile −7.1k, shock −15.0k). Window noise ≫ knob effects;
  a +500 window margin selects noise.
