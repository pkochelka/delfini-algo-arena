# Round 3 order-arrival processes

Tape: `data/round3.csv` (6000 s, 1,234,596 inserts, 56,274 IOC takers). Scripts are in this folder. Run with
`uv run --with pyarrow --with numba python marek/research/round3/flow/<script>.py` (`ladder_glm.py` also needs
`--with statsmodels`). Run order: `prep.py` (writes `cache/`, git-ignored), `bins.py`, `events.py`, `hmm.py`,
`regimes_coarse.py`, `ladder_build.py`, `shocks2.py`, then anything else. `dwell.py` and everything that uses S read
`../s_dynamics/s_path.parquet` (v0) and `../s_dynamics/vol_segments.csv`. `events.py` rewrites `shock_events.csv` and
`shocks3.py` extends it, so run them in that order.

## Summary: the arrival model

Round 3 is the round-2 arrival model with one new participant. The ladder maker is the whole "maker-rate episode" story.

1. **The same two-state Markov regime (calm/volatile), switching more often.** Calm dwell is about 154 s (r2 247), volatile
   dwell about 60 s (r2 43), and volatile occupancy is 0.28 (r2 0.15). The lead's high-maker minutes (10–13, 38–41, 63–68,
   92–98) are simply the volatile segments. There is no third state.
2. **A ladder maker, new.** It inserts runs of 1–6 orders on one side, one tick apart going away from the touch, with the
   rungs exactly **2.000 ms** apart. Its rate and size switch with the regime:
   - calm: about 6.6 events/s at ≈150 lots;
   - volatile: about 19.4 events/s at ≈33 lots.

   It produces the +34 makers/s and the larger median maker size in volatile minutes.
3. **The same 27 scheduled shocks.**
   - The drought lasts 3.0 s with maker inserts at ≈54/s. Ladders stop completely during it.
   - S jumps at the drought end (sd ≈ 47).
   - Takers burst afterwards. For the first 0.2 s the extra takers are all on the jump side; after that the burst is
     symmetric.
4. **Everything else is unchanged from round 2.**
   - Base makers: 180/s with round-2's size law.
   - Takers: 4/6 per side per s; coin-flip side; lognormal(4.09, 1.40) size; limit round(S ± 1) or round(S ± 2.55) plus
     an aggressive tail.
   - No Hawkes excitation, seasonality or informed takers.

| Component | Calm (72 % of time) | Volatile (28 %) | Shock (3 s drought, then burst) |
| --- | --- | --- | --- |
| Regime dwell | mean 154 s (95 % CI 110–232), exponential (KS p 0.67) | mean 60 s (CI 43–91), exponential (KS p 0.77), 4.5–249 s | independent of regime |
| Base makers | **180 /s**, side 50/50, round(lognormal(2.657, 1.803)) | 180 /s, same law | ≈ 54 /s for 3.0 s |
| Ladder events (incl. length-1) | ≈ 6.6 /s: 5.5 singles + 1.1 multi-rung | ≈ 19.4 /s: 8.2 singles + 11.2 multi-rung | **0** during the drought, ≈ 6 /s for a few s after the jump |
| Ladder rung size | ≈ 150 lots (71–320 band), median 145 | ≈ 33 lots (11–70 band) | — |
| Ladder inserts | ≈ 8 /s | ≈ 42 /s | — |
| All maker inserts | 188.5 /s | 222.5 /s | 53.7 /s |
| Takers per side | 4.0 /s (total 7.93) | 6.0 /s (total 11.97) | drought 8.7 /s total; jump-side +≈ 4 in the first 0.2 s, then +≈ 22 /s per side decaying linearly to baseline by +3 s |
| Taker side | coin flip (0.4985) | coin flip (0.502) | see burst |
| Taker size | rounded lognormal(4.094, 1.397), all states | same | same |
| Taker limit vs S | round(S ± 1): 94.5 %; tail 4.9 %, excess mean 3.4 | k = 2.55: 88.9 %; tail 9.3 %, excess mean 6.3 | — |

## Evidence

### 1. The "maker episodes" are the volatile regime (`bins.py`, `hmm.py`, `agree.py`, `volmk.py`)
- Per-minute maker counts run about 11,200 in calm and about 13,300 in episode minutes. Taker counts move with them,
  from ≈480 to ≈700 per minute.
- 1 s Poisson HMMs, with shock windows −2…+20 s treated as missing:
  - Taker-only, K = 2: 7.91 / 11.93 /s, the r2 values. BIC rejects K = 3 and K = 4.
  - Maker-only, K = 2: 188.5 / 222.5 /s.
  - The two agree on 97.7 % of seconds. The joint HMM agrees with the maker-only one on 99.2 %.
- Maker-only K = 3 and K = 4 split the volatile state into sub-states with dwell times of 2–3 s. That split comes from
  ladder clustering (below), not from a third regime.
- By size band, the volatile maker excess sits entirely in 11–70 lots (+38 /s). That band is overdispersed even at 0.1 s
  (DI 2.0). In calm the 90–320 band is overdispersed instead (DI 1.38 at 0.1 s). That pointed to batches.

### 2. The ladder maker (`pairs.py`, `ladder_gap.py`, `ladder_fine.py`, `ladder_exact.py`, `ladder_build.py`, `ladder_props.py`, `singles.py`)
- **The 2.000 ms clock.** For each insert, count the inserts that arrive exactly 1990–2010 µs later:
  - 45,072 sit at exactly 2000 µs. Each other µs bin holds ≈250, which is background.
  - Of these pairs, 99.2 % are on the same side and 95 % are one tick further from the touch. Nearly all are makers
    (IOC share 0.04 %).
- Chaining gives **23,892 multi-rung ladders holding 69k inserts**:
  - Lengths: 2 → 12.1k, 3 → 6.3k, 4 → 2.8k, 5 → 1.5k, 6 → 1.2k, 7 and longer → ≈ 50 (probably chance links).
  - Side split 49.6/50.4.
  - In 92 % of ladders every step is +1. The other steps are mostly 2–5 ticks.
  - One row per ladder: `ladder_events.csv`, with t0, side, n, p0, start depth, steps, sizes and regime.
- **Rung count is capped by depth.** The first rung sits at touch−2 in 61 % of volatile ladders; otherwise it starts
  between −3 and −6. A ladder never runs past touch−7: starting at −6 allows n ≤ 2, at −5 n ≤ 3, and so on. The makers
  agent owns placement, sizes and cancels; see their README.
- **Sizes.** Rung sizes within a ladder are strongly correlated (corr of log size between rung 0 and rung 1 is 0.84–0.88),
  so each ladder draws one size and adds noise per rung.
  - Calm: median ≈145 lots, log-sd 0.59.
  - Volatile: median ≈33 lots, log-sd 0.40.
- **Length-1 ladders hide among the regular makers.** Take the non-ladder makers per size band in deep regime interiors
  and compare them with 180 · P(band) under round-2's lognormal(2.657, 1.803):
  - The match is within ±0.3 /s in every band except the ladder bands.
  - Calm excess: +5.5 /s, all in 71–320 lots. Volatile excess: +8.2 /s, in 11–70 lots.
  - So the base maker process is exactly round 2's (180 /s), and the extra makers are all ladder orders.
- **Single regime clock** (`switch_align.py`). Aligned on S-based switch times, the ladder rate (1–2 → 9–12 /s), the rung
  log-size (4.6–5.0 → 3.45) and the taker rate (8 → 12) all flip within the same 1 s bin, in both directions. The base maker
  rate stays flat at ≈186 /s, which includes hidden singles.

### 3. Ladder arrival timing (`ladder_trigger.py`, `ladder_cond.py`, `ladder_touch.py`, `ladder_hazard.py`, `ladder_glm.py`, `ladder_xc.py`, `ladder_anchor.py`)
- **No clock and no fixed latency.**
  - The phase of ladder start times modulo periods from 1 ms to 1 s is flat (χ² p ≥ 0.09).
  - The time since the last taker, cancel, maker insert or row at ladder start is smooth, with no µs spike.
- **The hazard rises with recent price activity, symmetrically in both directions.** Per side, by the S move over the
  last 0.25 s:
  - Calm: 0.38 /s when S is still → 0.8 /s for 0.5–1.5 ticks → 3.6 /s for 1.5–3 → ≈7 /s for more than 3 ticks.
  - Volatile: 3.3 → 9 /s.
  - After any touch-price change, either side and either direction, ladder intensity on both sides is ×3–4 from lag 0
    (calm) and decays over 0.5–1 s. Volatile: ×1.4–1.7.
  - The effect starts at lag 0 and is not tied to the direction or side of the change, so this is a state-dependent hazard
    (recent activity), not an event trigger.
- **The regime matters on its own.** A Poisson GLM on 10 ms bins with regime plus an EWMA of touch changes (τ ≈ 0.25 s)
  still gives a regime coefficient of 1.6–2.0, i.e. ×5–7.
- **Self-excitation.** Ladder→ladder intensity is ×4.3 at 2.5–200 ms in calm and ×1.1–1.4 in volatile. Most of it is the
  shared activity driver. Counts are very overdispersed (DI ≈ 10 at 1 s).
- **Capacity.** In volatile, the hazard falls with the number of live ladder orders on that side: about 12 /s with 0–2
  live, against 2 /s with more than 20 live. This is probably capacity-limited, but it is confounded with recent activity.
- **Not reconstructed:** the exact trigger rule. A Poisson clock gated by some book or S condition fits everything above, but
  is not pinned down.

### 4. Regime switch times and dwell (`switch_viterbi.py`, `dwell.py` → `regimes.csv`)
- **Switch times** come from s_dynamics' S-based segmentation (`../s_dynamics/vol_segments.csv`, maker D law and taker
  offset given S): 28 volatile segments.
- **Independent check:** a 5 ms Viterbi using only flow evidence (taker arrivals, ladder starts, maker price vs touch,
  `cache/regimes_T0.1.csv`).
  - It recovers all 28 segments. Start and end times differ by a median of ≈ 0.3 s, at most 2.5 s (5543.0 vs 5540.5).
  - It also produces about 20 short 0.3–1.4 s "volatile" blips with LLR ≈ 20, just above the switching penalty. These are
    calm ladder clusters, not regime changes; the S-based segmentation has none.
- **Dwell laws, from `regimes.csv` (start, end, regime, dur, censored, shocks_inside):**
  - Volatile: n = 28, 4.5–249 s, mean 60.4 (95 % CI 43–91), CV 0.94, KS against exponential p = 0.77.
  - Calm: n = 27 uncensored, 3.8–572 s, mean 154 (CI 110–232), CV 0.86, KS p = 0.67.
  - Both look geometric (Markov). Exit rates are ≈ 0.0065 /s (calm) and ≈ 0.0166 /s (volatile).
  - Against r2's means (247 / 43), the likelihood-ratio test gives p = 0.02 (calm) and 0.057 (volatile). So the chain is
    the same form with faster calm exits.
- **Shocks and regime are independent.**
  - 5 of 27 jumps happen in volatile, against 7.6 expected.
  - No switch falls within 5 s of a jump; the nearest are 5.0 and 6.4 s.
  - Volatile entries within 30 s after a jump: 6, against 4.3 expected.

### 5. Shocks (`events.py`, `shocks2.py`, `shocks3.py`, `burst_*.py` → `shock_events.csv`)
- **Count and spacing.** 27 droughts. First onset 254.2 s, last 5955.2 s.
  - Gaps run 145.2–292.6 s (mean 219, sd 38).
  - Against U(150, 300): KS p = 0.46. Against exponential: p < 1e-4. Against 150 + Exp: p = 0.04, worse than in r2.
  - Two-sample KS against r2's gaps: p = 0.73. The pooled r2 + r3 gaps fit U(150, 300) with p = 0.23, but two r3 gaps
    (145.2, 145.5) and one r2 gap (147.0) fall below 150. So the law is probably U(≈145, 300) or similar.
- **Drought.** Lasts 3.03 ± 0.05 s, measurement included. Maker inserts run at 53.7 ± 4.9 /s, which is 0.29 × the base
  rate. **Ladder starts: 0 in 27 × 3 s**, against ≈ 30 expected even at the scaled rate.
- **Taker rate in the drought** is 8.7 /s, unchanged.
- **Jump time.** S jumps at onset + 3.032 ± 0.037 s, i.e. at the drought end (offset +0.003 ± 0.04 s).
- **Jump size.** J sd 47.2, mean |J| 34.5, excess kurtosis 0.9. Normal (KS p 0.48) and Laplace (p 0.35) are not
  distinguishable with 27 shocks. Range −111 to +124. r2 sd was 43.5.
- **No pre-shock tell.** Corr of J with the 10 s pre-onset maker rate, ladder rate, taker rate, spread and S drift, and
  with the 60 s S drift, is at most 0.27 in absolute value (n = 27). The same features against |J| are at most 0.36.
- **Post-jump taker burst (`burst_profile.py`), the same in r2:**
  - 0–0.2 s: jump-side takers arrive at ≈ 24–33 /s, about 4 extra per shock. Opposite-side takers stay at baseline.
    They are ordinary takers: limit ≈ round(S_new + 1), median size 40–90 lots, 80–100 % filled against the stale book.
    The first jump-side taker comes a median 77 ms after the jump (r2 70 ms).
  - From +0.2 s: symmetric, about 22 /s per side, decaying linearly to 4 /s by about +3 s. That is ≈ 30 extra takers per
    side, 66 ± 18 extra in total over 3.5 s.
  - Burst size does not scale with |J| (corr 0.02).
  - Because of the first 0.2 s, the share of burst takers on the jump side correlates with sign(J) (0.50 across shocks).
- **Makers after the jump.** Maker rate returns to normal at once. Ladder starts run at ≈ 6 /s for a few seconds, as the
  activity driver predicts.

### 6. Takers (`takers.py`, `taker_rule.py`, `taker_info.py`, `hawkes_check.py`)
- **Rates.** Calm 4.08 / 4.10 per side (buy / sell), volatile 6.23 / 6.17.
  - Per 1000 s block, calm runs 4.0–4.2 per side and volatile 6.0–6.6.
  - Within a regime, dispersion is 0.97–1.05 at 1, 10 and 30 s, so the process is homogeneous Poisson.
- **Size.** Rounded-lognormal MLE (4.0936, 1.3965); skew 0.000, kurtosis −0.02.
  - Tail: 1219 orders above 1000 against 1234 expected; 49 above 5000 against 43 expected.
  - The same in both regimes, on both sides and in bursts.
- **Side.** Sign ACF at lags 1–5 is at most 0.007 in absolute value. P(same side as the previous taker) is 0.50–0.52 for
  every gap size, so there are no meta-orders.
- **Limit vs S** (S v0, s_sd < 0.3):
  - Calm: best k = 1.00, 94.5 % in the band |x − 1| ≤ 0.5. The tail (4.9 %) has excess mean 3.4, median 2.5, 99th
    percentile 14.7, and is exponential-like (KS p 0.29).
  - Volatile: best k = 2.55, 88.9 % in the band. The tail (9.3 %) has excess mean 6.3, median 4.3, 99th percentile 30.
    That matches r2's 4–8 % tail of 0.5 + Exp(≈6) in mean; the share is at the top of r2's range.
- **Not informed.**
  - Corr(sign, past S move) over 0.5, 1, 5 and 30 s is at most 0.0094 in absolute value in both regimes (se 0.005–0.007).
  - Corr(sign, S − mid) is −0.001.
  - After the trade, E[sg·(S(t+h) − S_after)] is about +0.03 at 0.5–1 s in calm, then turns negative: −0.20 at 15 s and
    −0.28 at 60 s. That is the transient-impact decay novelty found, not information.
  - In volatile it is +0.2 at 0.5–1 s, then ≈ 0 by 30 s. The early positive drift is HMM smoothing; novelty saw the same
    on r2's S.
- **No Hawkes excitation among takers or base makers.** Intensity in the 0–1 s after an event, relative to the local
  baseline, is 0.96–1.08 for every pair. The only excitation is within the ladder streams: same-band M→M is ×1.8 in the
  first 5 ms, which is the 2 ms rungs.
- **Seasonality** (`misc.py`). Rates per 1000 s block are flat. The periodograms have no line: the largest peaks are 11–19×
  the median power at random periods, about the maximum expected from 3000 bins.

## Validation
- Per minute (`rates_60s.csv`): corr(vol_share, ladder starts) = 0.99 and corr(vol_share, takers) = 0.87–0.89. Base makers
  (non-ladder) correlate only 0.21 with vol_share, which is the hidden-singles excess of +2.6 /s in volatile.
- Base maker counts per size band match 180 · P_r2(band) to ±0.3 /s outside the ladder bands, in both regimes.
- Flow-only and S-only regime segmentations agree on all 28 segments.
- The burst profile and its jump-side head reproduce on r2 with the same numbers (`burst_profile.py`).

## Still open
- **The ladder trigger rule.** It is state-dependent (regime ×10, recent touch/S activity, live ladder count), with no clock
  and no latency. The start-depth and length law is with makers.
- **Hidden single-rung ladders.** Their count is inferred from size-band excess, not identified order by order. Size alone
  separates them only roughly (they share bands with base makers).
- **Calm dwell.** The mean is 154 s against r2's 247 (p = 0.02). The generator may use different chain constants, or the
  same form with the n = 27 noise.
- **Shock gap law.** U(145–150, 300) against alternatives, and normal against Laplace for J. With n = 27 per round these
  cannot be separated; pooling r2 and r3 (n = 54) still doesn't settle it.
- **The jump-side taker head in 0–0.2 s.** Is it a separate burst component (≈ 4 takers at ≈ 25 /s), or side =
  sign(S − stale book) for burst takers while the book is stale? The S − mid test can't tell, because makers re-quote at
  the new level within ms.

## Trading implications
1. **The regime is easier to read than in r2.** Ladder inserts (exact 2 ms same-side chains) jump from about 1 to about
   11 starts/s in volatile, and their rung size drops from ≈150 to ≈33 lots.
   - Two or three ladders within 0.5 s, each with ≈30-lot rungs, signal volatile within about 0.3 s.
   - The r2 tells still work: spread > 1.5; maker quotes never within 0.7 of S.
   - Volatile is now 28 % of the time with ≈ 60 s dwell, so the volatile handling matters about twice as much as in r2.
2. **Ladder liquidity sits 2–7 ticks behind the touch.** Volatile ladders are small (≈ 33 lots/rung). Calm ladders are
   large (≈ 150/rung) but only appear after price activity. They are cancelled once the touch is more than 7 away (makers).
   Price impact from fills against ladders counts like any other fill (novelty).
3. **Shocks are unchanged.**
   - Detection: fewer than 50 maker inserts in 0.5 s, plus **no ladder starts**, which is an extra confirming tell.
   - Pull quotes before onset + 3.0 s.
   - During the first 0.2 s after the jump, ≈ 4 extra takers per shock hit the stale side in the jump direction. Don't
     have resting quotes there.
   - From +0.2 s the burst is symmetric (≈ 22 /s per side for ≈ 1 s), so quote both sides at the new level.
   - The next shock comes no sooner than about 145 s later.
4. **No directional alpha in taker flow.** The one predictable piece is the decay of each taker's own impact
   (transient, ≈ 50 % over 15–40 s; see novelty and s_dynamics).

## Files
- `regimes.csv`: final calm/volatile segments (S-based switch times), with dur, censored flag and number of shocks inside.
- `regimes_1s.csv`: the coarse 1 s joint-HMM segmentation (posted early).
- `cache/regimes_T0.1.csv`: the flow-only 5 ms Viterbi.
- `shock_events.csv`: one row per shock. Columns: onset, drought end/dur/maker rate, t_jump, jump lag, J (from S v0),
  regime at jump, drought taker rate, burst count and buy share, first jump-side taker latency, pre-onset features,
  post-60 s move.
- `ladder_events.csv`: one row per multi-rung ladder.
- `rates_60s.csv`: per-minute rates by stream, with volatile share.
- `cache/ladders.pkl`, `cache/ladder_member.npy`: ladder table and per-insert membership (insert index → ladder id, rung).
