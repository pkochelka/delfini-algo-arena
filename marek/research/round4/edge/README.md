# Round 4 edge pass: what is left after r4_volsig

Written 2026-11-08. Tape `data/round4.csv` only, one regression run on `data/round3.csv`. Base
`strategies/marek/round4/r4_volsig.py` (1,413.0k: calm 281.0k, volatile 821.6k, shock 310.4k). Dev file
`strategies/marek/round4/r4_v3_dev.py` = r4_volsig + the v2 ladder fix (`lad_hist`, `lad_step`) and velocity-prior knobs
(`kv_inf`, `kv_shr`, `kv_m`) ported from r4_v2_dev + this pass's `tk_free`, all off by default (= r4_volsig, 1,412,970
reproduced). Harness copies (`eval4.py`, `sweep.py`, `summ.py --baseset`, `tbl.py`) write to `results.jsonl` here.

**Bottom line.** No edge is left inside this architecture on round 4, and the ceiling is now measured rather than
argued. (1) Every one of 43 single-knob changes to the calm, volatile and take rules lands at or below the base on
calm + volatile (cv), so the round-3 tuning is at its optimum on round 4 and the base path itself carries ≈ 4k of
selection luck. (2) Replacing the fair value inside the quoting logic by the *smoothed true S path* (look-ahead) adds at
most +17.5k cv (1.6 %); the research group's semi-causal reference filter adds 0. The S filter is not the limit.
(3) Momentum beyond the filter's own state is worth ΔR² ≤ 0.007 at 0.5–5 s in volatile and ≈ 0.01 in calm; states where
the pinned position sits against a strong velocity are 2.7 % of volatile time and the move they predict over the flip
lag is about the crossing cost. (4) The one mechanical slack found, IOC takes truncated by the room our own resting
quote holds (38k lots in calm), loses −8..−10k when freed, because the cancelled quote loses its queue position.
The limit is the ±100 cap and the flow the tape offers, as the earlier passes concluded.
**Recommended file: `r4_v3.py` = r4_volsig + the ladder-filter fix (lad_hist 1, lad_step 8) + kv_inf 1, kv_shr 1**
(see "Final" for the numbers; the gain is noise-sized, the fix is kept for filter correctness).

## Tools

| Path | What |
| --- | --- |
| `tools/logx.py` | one run -> `cache/<strat>_round4.npz`: 10 ms state (F, sd, pv, turb, pos, touch, Y, kv, drought flag), every market taker (t, side, qty, levels), every order we send (side, price, qty, IOC, pos, resting / cancelling same-side qty, turb, shock, sniper, stale qty at the IOC limit, F, sd), every fill (side, price, qty, IOC, pos before, F), PnL curve |
| `tools/mom.py` | momentum as an increment over the filter state: OOS R² of S(t+h) − F(t) on [kv, F − mid, Y] with and without taker-flow / past-move tells; position alignment with the next move |
| `tools/trans.py` | regime transitions: turb detection lag vs `s_dynamics/vol_segments.csv`, PnL around starts / ends, turb-vs-truth confusion |
| `tools/pinned.py` | PnL per second by \|pos\| bucket and regime; 1 s carry; IOC takes vs the room held by our own resting quotes |
| `tools/episodes.py` | take episodes (same-side IOCs < 100 ms apart): is the room truncation recovered by follow-up takes? |
| `tools/takes.py`, `tools/takes2.py` | fill markouts vs s_path by (regime, IOC, reducing) and by F-edge bucket; take limit edge |
| `tools/ferr2.py` | F − S error of logged runs by regime |
| `tools/oracle.py` | ceiling run: F inside `_quote` replaced by `s_path` `s_filt` (semi-causal reference) or `s_hat` (smoothed); rows tagged `oracle_*` |

## 1. Where the PnL comes from (r4_volsig, shock windows −5..+25 s removed)

PnL per second by |pos| (`pinned.py`):

| regime | \|pos\| 0–20 | 20–50 | 50–80 | 80–95 | 95–100 | pinned share |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| calm | 33.5 /s | 38.9 | 62.7 | 59.7 | **90.2** | 54 % |
| volatile | 117 /s | 143 | 228 | 251 | **562** | 89 % |

- Being pinned is the *good* state in both regimes. Calm 1 s carry is +9k of 256k (calm PnL is spread, not drift);
  volatile carry is 496k of 804k, 491k of it while pinned.
- Fill markouts vs S (lots-weighted, ticks per lot; `takes.py`):

| regime | kind | lots | at fill | +1 s | +5 s | +20 s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| calm | passive, increasing \|pos\| | 159k | +0.64 | +0.15 | −0.01 | −0.16 |
| calm | passive, reducing | 581k | +0.68 | +0.16 | +0.15 | +0.27 |
| calm | IOC take, increasing | 121k | +0.39 | **+0.89** | +1.12 | +1.08 |
| calm | IOC take, reducing | 133k | +0.22 | +0.64 | +0.85 | +0.11 |
| volatile | IOC take, increasing | 55k | +0.69 | +3.60 | +6.76 | +4.25 |
| volatile | IOC take, reducing | 106k | +0.42 | +2.89 | +4.41 | +6.16 |
| volatile | flatten quote (vf) | 6k | +2.19 | +1.24 | +6.45 | +9.95 |

- Calm passive fills are worth ≈ 0 by 5 s (the taker's own kick plus the side-law continuation eats the 0.64 edge);
  the calm money is in the IOC takes, which gain +0.5 *after* the fill (continuation), in every F-edge bucket
  (`takes2.py`: even takes at 0–0.25 edge mark out +0.24 / +0.79 at 1 / 5 s). Yet lowering the calm take threshold
  loses in the engine (below): the takes are room-limited 52 % of the time, not threshold-limited.
- Regime detection is not a problem: P(turb | volatile) 0.991, P(turb | calm) 0.006, start lag median 0.53 s, and the
  first 3 s of a volatile episode earn 700 /s (above the 514 /s average), because V leads the switch (`trans.py`).
- Pre-jump position: mean |pos| 71 at onset + 2.9 s, so each shock carries a zero-mean lottery of sd ≈ 5.5k
  (+6.4k net luck on this tape). Not an edge; the snipe itself is at its visible-depth ceiling (jumps pass).

## 2. Hypotheses and their tests

| # | Hypothesis | Test | Result |
| --- | --- | --- | --- |
| H1 | Tape-only momentum tells (taker-flow imbalance over 0.25–2 s in lots / impact units / counts, past mid or F moves) add to the filter's kv and F − mid | `mom.py`: OOS half-split R² of S(t+h) − F(t), volatile | base R² 0.33 / 0.38 / 0.37 / 0.23 at 0.5 / 1 / 2 / 5 s; **+0.003..+0.007** with every tell set; tells alone 0.26. Calm: ≤ 0.02 everywhere. Dead (confirms the earlier lean results) |
| H2 | Flipping aggressively when the pinned position is against a strong kv pays | table of E[dS · sign(pos)] by signed kv bucket, volatile, \|pos\| ≥ 80, 50 ms samples | against kv ≤ −2: **2.7 % of time**, next 1 s move −2.7..−4.8 /lot, crossing cost (S − bid) 0.8–1.0; the strategy flips on the next stale quote ≈ 0.3–0.5 s later anyway, so the gain over that lag ≈ the cost. Aligned kv ∈ [0, 1) already carries +2.7 /lot/s. Dead; vs_kv 0.15 is the right size |
| H3 | Filter precision is a lever (F − S rmse 0.41 volatile vs the reference causal filter's 0.25) | `oracle.py`: F inside `_quote` := s_filt / s_hat, thresholds loosened for a perfect F; `ferr2.py` with the ladder fix | **ceiling +17.5k cv** with the smoothed path (look-ahead), reference filter 0..−13k; ladder fix rmse 0.407 → 0.386 and cv −2.7k. Closed |
| H4 | Takes truncated by the room our own resting same-side quote holds leave stale depth on the table | `pinned.py`, `episodes.py`; knob `tk_free` (cancel the resting quote and take with its room) | volatile: follow-up takes within 100 ms recover all but 3k lots. Calm: 38k lots short (946 of 3,310 takes have a 92-lot resting quote on the same side); in the engine `tk_free` 1 / 2 / max 100: **calm −8.2 / −9.4 / −8.8k** (the cancelled quote is re-posted at the back of the queue). Dead |
| H5 | The calm and volatile quoting / take parameters tuned on round 3 are off-optimum on round 4 (calm now has the side law, V and the 15-lot drift) | 43 single-knob runs (`cfg/calm1.jsonl`, `cfg/turb1.jsonl`) | **all 43 at or below the base on cv** (best −0.1k vf_keep 1.0, −0.8 theta_keep 0.05, −0.9 theta 0.35); the base path is ≈ 4k lucky. Big losers: reach_off 1.5 −64k, passive_t 1 −93k, vf_mom off −33k, inv_skew 0.3 −23k, take_t 0.4 −17k, take_sd_t 1.5 −14k |

## 3. Engine results (deltas vs r4_volsig in k; cv = calm + volatile, shock column is #6 / #24 lottery)

| Knob | Total | calm | vol | cv | shock |
| --- | ---: | ---: | ---: | ---: | ---: |
| lad_hist 1, lad_step 8 (v2 ladder fix) | 1436.8 | −0.6 | −2.1 | −2.7 | +26.5 (#6 +16.9) |
| + kv_inf 1, kv_shr 1 | 1433.1 | −0.7 | +2.9 | +2.2 | +17.9 |
| take 0.05 / 0.15 / 0.2 | 1421.6 / 1415.0 / 1411.2 | | | −3.2 / −2.2 / −1.1 | |
| take_sd 0.5 / 1.5; tr_take_sd 0.5 / 1.5 | | | | −4.1 / −2.0; −3.9 / −1.7 | |
| theta 0.2 / 0.3 / 0.35; theta_keep 0.05 / 0.1 | | | | −2.0 / −5.6 / −0.9; −0.8 / −4.6 | |
| inv_skew 0.1 / 0.2 / 0.3 | | −4.4 / −13.3 / −21.3 | | −3.7 / −13.9 / −22.7 | |
| size_top 50 / 100; max_sd 0.4 / 0.8; keep_depth 2; reach_off 1.5 | | | | −4.8 / −3.0; −2.1 / −1.8; 0; −63.9 | |
| tk_free 1 / 2 / (1, max 100) | 1417.0 / 1416.5 / 1412.5 | −8.2 / −9.4 / −8.8 | +0.3 / +0.3 / −1.3 | −7.9 / −9.1 / −10.2 | |
| take_t 0.2 / 0.4; take_sd_t 0.5 / 1.5 | | | −0.7 / −16.0; −3.1 / −12.2 | −2.7 / −16.8; −6.6 / −13.5 | |
| vf_th 1.5 / 2.5; vf_keep 1.0 / 2.0; vf_pos 40 / 80; vf_q 200 | | | | −2.3 / −7.5; −0.1 / −2.1; −5.1 / −3.0; 0 | |
| vf_mom −0.25 / −1.0 / off | | | −0.1 / −5.5 / −32.1 | −1.8 / −7.3 / −32.7 | |
| reach_t 2.0 / 3.0; pv_thr 0.5 / 0.85; passive_t 1 | | | | −6.5 / −3.9; −5.1 / −3.2; −93.2 | |
| oracle F = s_hat (smoothed); + take_sd 0 / take_sd_t 0; + theta 0.15 / theta_t 1.0 | 1405.2 / 1402.4 / 1402.1 | +13.7 / +12.2 / +10.9 | −2.6 / +5.3 / +5.8 | +11.0 / **+17.5** / +16.6 | (sniper disturbed) |
| oracle F = s_filt (reference); + its sd | 1363.2 / 1392.0 | −1.9 / +2.8 | −11.2 / −2.7 | −13.2 / 0.0 | |

## 4. Final

`strategies/marek/round4/r4_v3.py` = r4_volsig + `lad_hist 1, lad_step 8, kv_inf 1, kv_shr 1` (same imports, no file
reads). Round 4 1,433.1k (calm 280.3k, volatile 824.5k, shock 328.3k; cv +2.2k vs r4_volsig, noise; shock +17.9k is
#6 lottery). Round 3 850.0k (cv +6.2k).

| Run (tags `base`, `final`, `r3reg`) | Total | calm | volatile | shock | breach |
| --- | ---: | ---: | ---: | ---: | ---: |
| r4_volsig, round 4 | 1,413.0k | 281.0k | 821.6k | 310.4k | 12.0k |
| **r4_v3, round 4** | **1,433.1k** | 280.3k | 824.5k | 328.3k | 13.7k |
| r4_volsig, round 3 | 845.4k | 254.0k | 419.1k | 172.2k | |
| **r4_v3, round 3** | **850.0k** | 253.6k | 425.7k | 170.7k | |

- Round 4 cv +2.2k (ladder fix alone −2.7k, kv_inf / kv_shr on top +4.9k); round 3 cv +6.2k, up in 8 of 12 windows,
  t_cv 2.3 (ladder fix alone +3.6k, 7 of 12). Both are inside the ≈ 4k path-luck band of the base, so the file is kept
  for the measured filter correction and the consistent sign on two tapes, not as a proven gain.
- `r4_v3_dev.py` with every knob off reproduces r4_volsig exactly (1,412,970); `r4_v3.py` = dev + the final update.

**What would move the needle** (not available inside this architecture): more position room (the cap), or a predictor
of 1 s moves far better than the filter's R² 0.38; the oracle run shows that even the true S path only buys 1.6 %.

## 5. Interpretability: what r4_v3 earns its money with, and a small readable model

Tools added for this pass: `tools/ferr_kv.py` (F − S error by |kv|: trend lag), `tools/shockdump.py` (order-by-order
sniper dump per shock), `tools/firelog.py` (every sniper fire decision and the market events around it),
`tools/readings.py` (which inserts feed the tracker, joined to the true per-insert class), `cfg/simple_ablate.jsonl`.

### 5.1 Ablation of r4_v3 (tag `ablate`, round 4, deltas vs 1,433.1k)

| Rule turned off | Δ total | Δ calm | Δ vol | Δ shock | What the rule is |
| --- | ---: | ---: | ---: | ---: | --- |
| volatile IOC takes (`take_t 0`) | −956k | −19 | −825 | −112 | IOC every stale quote priced through F by > 0.3 + sd |
| regime HMM (`regimes 1`, calm laws only) | −533k | −14 | −415 | −104 | makers quote 3× wider in volatile; without it F is wrong by ticks |
| calm passive quotes (`min_post 1000`) | −215k | −140 | −7 | −69 | rest the whole room at F ∓ 0.25 |
| calm IOC takes (`take 0`) | −145k | −116 | −3 | −26 | IOC stale quotes through F by > 0.1 + sd |
| shock sniper (`ss_on 0`) | −144k | 0 | −2 | −142 | IOC the stale side in the 3 s drought window |
| oversize (`xs_cap 0`) | −128k | +1 | 0 | −129 | first shot up to 950 lots when the jump is known to be ≥ 4 ticks |
| ladder filter (`lad_on 0`) | −101k | −18 | −61 | −22 | drop the 2 ms rung chains from the filter |
| velocity state (`kv_on 0`) | −72k | −1 | −43 | −27 | S has momentum; F = S + 0.15·V |
| eta 0.15 → 0.005 | −49k | −30 | +1 | −20 | outlier share of maker inserts in the likelihood |
| vf flatten quote (`vf_on 0`) | −34k | −1 | −4 | −28 | volatile reduce-only quote 2 ticks beyond F |
| vs_kv nowcast (`vs_kv 0`) | −31k | +2 | −23 | −10 | lean F by the velocity |
| drought quoting (`dr_on 0`) | −28k | +1 | −1 | −29 | keep passive quotes during the drought |
| reduce-take (`tr_take null`) | −11k | +1 | 0 | −12 | calm: IOC the position down at edge ≥ sd |
| transient Y (`tr_on 0`) | +1k | −1 | −1 | +3 | useless on round 4 |

Reading: four things matter — volatile takes against stale quotes (the regime detector exists to make them possible),
calm passive quotes + calm takes (≈ 360k), the sniper with oversize (≈ 270k) and the ladder filter. The 481-cell grid
posterior, the transient-impact state and the filter's many small corrections are worth ≈ 0–70k each.

### 5.2 The small model: `strategies/marek/round4/r4_simple.py` (≈ 560 lines, 15 s per run)

One rule per generator fact (docstring of the file): a 2-state Kalman filter (S, V) fed by every new market quote as
`S ~ P ± (0.5 + d0 + λ)` with noise λ² + 1/12 and by taker kicks c·q^e; the regime weight is an EWMA of the observed
quote offset (w_lo 0.8 … w_hi 1.4 ticks); a gate, a two-reading jump re-anchor; the ladder rule (rungs 2000 µs apart
are dropped, a quote 2+ ticks behind the touch is held 2 ms for a rung); calm quotes at F ∓ 0.25 with takes at
0.1 + sd and reduce-takes at sd; volatile = takes at 0.3 + sd plus the reduce-only flatten quote; the drought sniper
with a clip trigger from onset + 2.93 s, far-quote / sweep triggers from onset + 2.99 s, 3 shots, 950-lot oversize
when the trigger proves ≥ 4 ticks, excess unwound after the shots at F ∓ 1 (then F ∓ 4).

| Step (tags `simple1`, `simple2`, `simple_sw`, `simple_sn`, `simple_lad`, `simple_calm`, `simple_final`) | Total | calm | volatile | shock |
| --- | ---: | ---: | ---: | ---: |
| v0: tracker + quotes + takes + sniper, no oversize | 1,181.3k | 228.9k | 796.5k | 155.9k |
| + oversize `xs_cap 950` | 1,189.2k | | | 163.7k |
| + clip trigger carries the oversize (`clip_edge 4`) | 1,275.2k | | | 249.8k |
| + far triggers only from 2.99 s, unwind after the shots (`ss_w0`, `xu_wait`) | 1,299.7k | | | 274.3k |
| + ladder-start hold (`lad_deep 2`) | 1,337.1k | 233.6k | 798.3k | 305.3k |
| + regime EWMA half-life 0.15 s (`w_hl`) | **1,350.1k** | 236.4k | 807.0k | 306.7k |
| r4_v3 | 1,433.1k | 280.3k | 824.5k | 328.3k |
| oracle: simple model with the true S inside `_trade` | 1,244.5k (v0) | 281.4k | 820.3k | |
| **round 3**: simple final / r4_v3 | 743.7k / 850.0k | 211.6k / 253.6k | 427.7k / 425.7k | 104.4k / 170.7k |

So the readable model reaches 94 % of r4_v3 on round 4 (87 % on round 3): volatile within 2 % on both tapes, shock
within 7 % on round 4, calm 16 % behind on both. The oracle row says the whole calm + volatile gap is tracker error
(with the true S the simple quoting/take rules earn exactly what r4_v3 earns), and the shock gap is sniper mechanics.

**How each shock-side fix was found** (`shockdump`, `firelog`): the big model's money on the large jumps (#6 +76k,
#23 +35k, #0 +31k) is one 850-lot shot sent 7–25 ms *before* the jump lands, on the informed sweep / clip that
precedes it; the simple model fired the same trigger but with the plain 100-lot room → clip edge credit. Then the
far-quote trigger fired the wrong way at #1, #2, #20 on the drought's own 3-tick tail quotes 30–80 ms before the
jump → far triggers only from onset + 2.99 s (the big model's `ss_w0`); and at #14 the excess was bought back at the
stale ask 13 ms before the jump because F still sat at the old level → unwind only after the shots.

**Tracker error** (`ferr_kv`, `readings`): the simple tracker's volatile rmse was 0.77 vs 0.41 and grew with |kv|
(1.6 ticks at |kv| ≥ 6, lagging S), with the error persistent (ACF 0.42 at 0.5 s) and pulled toward the mid (F − mid
< −2 → F 1.0 too high). 99 % of its readings are real base-maker quotes with the generator's residual sd, so the
contamination was the ladder chains' *first* rung (placed relative to the stale touch, not S): holding deep quotes
2 ms (the big model's `lad_deep`) cut the volatile rmse to 0.70 and the |kv| ≥ 6 lag to 0.76, but moved the volatile
PnL by only −2k (the oracle says all of the volatile error is worth 24k). Faster velocity (`v_sd_v 6`), a wider gate,
a different half-tick (`y_half`), reading noise ×0.6 / ×1.7, more level noise: each trades trend lag against quiet
noise, none better overall. The remaining gap (0.70 vs 0.41 volatile, 0.18 vs 0.14 calm) is the Gaussian reading
model against the exponential maker law: the grid posterior knows a quote near S is the *likely* one and a far
quote is a tail, the Kalman filter treats both as ±1 tick evidence. Windowed hard bounds (`bound_win`) and an
asymmetric tail noise (`tail_k 3`: calm rmse 0.22 → 0.18, but calm PnL −6k and shock −38k) do not fix that.

**Calm**: both models earn the same per second in each |pos| bucket (pinned 90 /s vs 89 /s); the simple model is
just pinned 32 % of calm time instead of 54 % — its pinned episodes last 0.5 s instead of 0.7 s because the
reduce-side quote at F ± 0.25 is placed a bit toward the mid (F bias −0.1 when S is 1 tick beyond the mid) and gets
hit sooner. Turning the reduce-take off: −2.6k; thresholds 0.5·sd / 2·sd: −8k / −62k cv.

### 5.3 Ablation of the simple model (tag `simple_abl`, round 4, deltas vs 1,350.1k)

| Rule turned off | Δ total | Δ calm | Δ vol | Δ shock |
| --- | ---: | ---: | ---: | ---: |
| regime: always calm (`w_lo 100, w_hi 101, jump_hold 0`) | −1,260k | −28 | −1,133 | −100 |
| volatile takes (`tau_t 100`) | −882k | −20 | −769 | −94 |
| regime: always volatile (`w_lo −2, w_hi −1`) | −310k | −302 | 0 | −9 |
| drought detection (`dr_n 0`, no shock handling at all) | −193k | 0 | 0 | −193 |
| sniper shots (`ss_shots 0`) | −191k | 0 | 0 | −191 |
| calm passive quotes (`max_sd 0`) | −122k | −99 | +1 | −23 |
| oversize (`xs_cap 0`) | −113k | 0 | 0 | −113 |
| edge credits (`sw_edge 0, clip_edge 0`: oversize only when the tracker already sees ≥ 4 ticks) | −107k | 0 | 0 | −107 |
| velocity state (`v_sd 0, k_now 0`) | −90k | −7 | −50 | −33 |
| ladder rule (`lad_on 0`) | −63k | −3 | −51 | −8 |
| calm takes (`tau_c 100`) | −58k | −47 | 0 | −10 |
| velocity nowcast only (`k_now 0`, F = S) | −44k | −1 | −18 | −25 |
| sd term in the take threshold (`tau_sd 0`) | −15k | −16 | +5 | −4 |
| jump re-anchor (`far 1000`) | −7k | 0 | −4 | −3 |
| volatile flatten quote (`flat_pos 1000`) | −4k | 0 | −5 | +1 |
| bound clamp (`bound 0`) | +1k | −1 | +2 | 0 |
| gate (`gate 100`) | +19k | −13 | +4 | +28 |

Same ranking as the big model's: the regime switch and the volatile takes are the business, then calm quotes + takes,
then the sniper (whose value is almost entirely the oversize carried by the first shot), then velocity and the ladder
rule. The gate, the bound clamp and the jump re-anchor are cosmetic (the gate's +19k is the shock lottery; it costs
13k calm, kept for the calm side).

**Verdict.** A model whose every line can be read earns 94 % of r4_v3 on round 4 and 87 % on round 3, matching it in
the volatile regime and (on round 4) in shocks. What the big model's extra machinery buys is one thing: a sub-tick S
estimate in calm from the exact exponential maker likelihood (≈ 45k calm + 17k volatile on round 4, ≈ 42k calm on
round 3). The transient-impact state, the HMM's exact transition structure and the many filter corrections buy
nothing measurable. For the competition keep `r4_v3.py`; for explaining what the strategy does, `r4_simple.py` is the
document.
