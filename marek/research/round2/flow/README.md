# Round 2 order-arrival processes

Tape: `data/round2.csv` (6000 s, 1,123,282 inserts, 1,069,782 cancels). All scripts are in this folder and run with
`uv run python marek/research/round2/flow/<script>.py`. Run `prep.py` first: it writes `cache/` (git-ignored, about 320 MB).

## Summary: the arrival model

The round-1 generator is still underneath. Three things are layered on top of it:

1. **A two-state Markov regime, calm or volatile.** It modulates taker rate, price impact, quote width and taker
   aggressiveness. Maker rate does not change.
2. **Scheduled "shock" events, 27 in 6000 s.** In each one, maker inserts drop to 30 % for exactly 3.0 s. S then jumps
   by a random amount (sd ≈ 43 ticks). A symmetric taker burst lasting about 3 s follows.
3. Nothing else. Within a regime and outside shocks, every stream is a homogeneous Poisson process: dispersion 0.96–1.04
   at 1 s and at 10 s. There is no self-excitation, no seasonality, no informed taker side and no special large orders.

| Component | Calm (≈85 % of time) | Volatile (≈15 %) | Shock (3 s drought, then 3 s burst) |
| --- | --- | --- | --- |
| Maker rate | 180.0 /s, side 50/50 | 179.9 /s | **54 /s (×0.30) for 3.00 s**, then normal |
| Taker rate per side | **4.00 /s** (3.99 / 4.00) | **6.0 /s** (5.99 / 6.18) | ≈4 /s in the drought. After the jump: about +35 /s total for 1.25 s, falling linearly to baseline by +3 s (≈80 extra takers, + ≈0.5 per tick of \|jump\|) |
| Taker side | coin flip | coin flip | coin flip (burst buy share 0.49) |
| Taker size | rounded lognormal(4.10, 1.40) | same (log-mean 4.12) | same (4.16 ± 0.03) |
| Maker size | rounded lognormal(2.657, 1.803) | same | same |
| Impact (10 s bins, dmid ~ c·Σ sign·√filled) | c ≈ 0.030, R² 0.80 | **c ≈ 0.13**, R² 0.93 | exogenous jump |
| Mid vol, 1 s | sd ≈ 1.1 tick | sd ≈ 5–6 tick | — |
| Maker envelope width (min ask − max bid of inserts in 25 ms) | 1.70 | **3.40** | wide |
| Taker limit − maker-envelope centre | 1.0 (round-1 rule round(S ± 1)) | **≈2.5–3.0** | — |
| Book spread (time-weighted, per s) | 1.07 | 1.86 | thin book (touch qty 113 vs 880) |
| Maker lifetime, mean | 0.42 s | 0.33 s | **0.19–0.20 s** for orders placed in the drought |

**Regime switching.** The regimes come from a 2-state Poisson HMM on 1 s taker counts, with shock windows treated as
missing. BIC picks K = 2; K = 3, 4 and 5 are all worse.
- Calm → volatile: p = 0.0040 /s, mean dwell ≈ 250 s.
- Volatile → calm: p = 0.023 /s, mean dwell ≈ 44 s.
- Segment lengths: volatile 7–229 s (17 segments, sd ≈ mean, so consistent with geometric); calm 14–1022 s.
- A second HMM on maker-envelope width gives 98.2 % agreement. The book reveals the regime almost immediately:
  per-second spread > 1.5 classifies 96.7 % of seconds correctly.

**Shock schedule.** Onsets fall at 270.9, 443.9, 661.4, … 5842.7 s. Gaps between onsets run 147–296 s (mean 214, sd 46).
- They are not Poisson: the KS test against exponential gives p < 1e-4.
- U(150, 300) fits with p = 0.44, as does 150 + Exp(64) with p = 0.33. The smallest gap (147.0 s) sits slightly below 150.
- The drought lasts 3.03 ± 0.06 s, measurement included, so it is a fixed 3.0 s.
- The jump comes at the drought end (+3.0 s). Makers start quoting at the new level instantly, with the stale book still
  resting. That makes it an exogenous S jump, not one driven by takers.
- Jump sizes: +11.5, +12, −5.5, −6, −12.5, −17.5, +52.5, −20, −3.5, −33.5, +35.5, −55.5, +29, +31.5, −54.5, −24, −88.5, −12,
  −51, +22.5, +10.5, +47.5, +5, +1.5, +114.5, +100, −3.5. Mean \|J\| = 32, sd = 43.5, excess kurtosis 0.7.
- Nothing seen during the drought predicts the jump's direction: maker side or volume imbalance, envelope drift, drought
  taker flow, or mid drift (\|corr\| < 0.3, n = 27). The prior 60 s move correlates +0.42, which is weak evidence with n = 27.
- Shocks don't line up with the volatile regime. Volatile occupancy around shocks is 0.08–0.19, against 0.147 overall.

## Evidence (by question)

### 1. Rates over time (`rates.py`, `acf.py`, `xcorr_fine.py`, `events.py`, `residual.py`, `hmm.py`, `regimes.py`)
- **Raw tape.** Dispersion at 10 s is 2.6 for makers and 3.2 for takers. In 10 s bins, maker and taker counts correlate
  −0.38. The cross-correlation corr(maker[t], taker[t + l]) reaches −0.084 at l = +3 s in 0.1 s bins. That is the
  drought-then-burst signature, and it led to the events.
- **Makers with shock windows removed** (−2 s to +20 s around each onset). Dispersion is 0.98 at 1 s and 0.95 at 10 s,
  and the ACF is about 0 at every lag. Makers are homogeneous Poisson at 180 /s, and the regime doesn't affect them.
  - The 60 s dispersion of 0.68 is slightly low (about −2σ), and no cause was found.
  - The gap KS test against exponential gives p = 0.86, CV 0.999.
- **Takers with shock windows removed.** Dispersion is 1.22 at 1 s, 3.1 at 10 s and 10 at 60 s. The slow ACF (0.21 at
  1 s, 0.11 at 30 s, about 0 at 120 s) comes entirely from the regime. Within a regime, dispersion is 0.96–1.02 at 1 s
  and at 10 s.
- **Hawkes test** (`hawkes_check.py`). After any event, the intensity of every stream (TB, TS, MB, MA) in the next
  0–10 ms, 10–100 ms, 0.1–0.5 s and 0.5–2 s is 1.00–1.05 times its local baseline. The same small, uniform excess shows
  up for every pair, so it comes from the baseline estimate. **There is no self- or cross-excitation.**
- **Volatility coupling.** In the calm regime, taker count does not depend on realised volatility: 77–81 per 10 s across
  rv deciles 1–8. Counts jump to 103–118 only in the top two deciles, which are volatile-regime bins. The rate is
  regime-switched, not volatility-driven.
- Per-segment table: `taker_regimes.csv`. The sharper envelope-width segmentation is in `regimes_envW.csv`; it also tags
  the 3 s droughts.

### 2. Takers (`takers.py`, `taker_info.py`, `impact_regime*.py`, `segment_impact.py`, `taker_rule*.py`, `offsets_*.py`)
- **The side is still a coin flip.** Buy share 0.4987. Sign ACF at lags 1–5 is within ±0.006. P(same side as the
  previous taker) is 0.49–0.51 at every gap size, so there are no meta-orders. Corr(sign, past 1/5/30 s mid move) is
  within ±0.007 in calm and volatile alike. In volatile segments, buy share is unrelated to the segment's net move: the
  −270-tick segment has 0.493.
- **Future move after the trade**, E[sign·(mid(t+h) − mid_post)] at h = 0.1 / 1 / 5 s:
  - Calm: +0.054 / 0.206 / 0.226.
  - Round 1: +0.053 / 0.198 / 0.198, essentially the same.
  - Volatile: +0.50 / 0.95 / 1.11.
  - The volatile excess is impact, not information. It scales with √size, regressed on **ordered** size (exogenous) as
    well as filled size. Slope per √ordered: 0.023 in round 1, 0.029 in calm, **0.117 in volatile**.
- **Impact.** In 10 s bins, dmid = c · Σ sign·√filled gives c = 0.030 (R² 0.80) in calm and c = 0.133 (R² 0.93) in
  volatile.
  - Regime-switched impact of about ×5 plus the shock jumps probably explains most of the "exogenous" S motion. Calm-regime
    residual sd is 1.7 ticks per 10 s, so some extra calm diffusion is possible. This is for the s_dynamics study to confirm.
  - Caveat: filled quantity is endogenous in the volatile regime, because a stale book fills deeper on the S side. Ordered
    size gives c ≈ 0.115.
- **Limit price.**
  - Price minus the touch: 0 or +1 ticks for 90 % of calm takers. In the volatile regime, 48 % go ≥ 2 ticks through the
    touch.
  - Against the envelope of maker inserts in the 30 ms before the taker: calm offset 1.00 (median 1.0, same shape as
    round 1, plus a ≈3 % right tail), volatile offset ≈ 2.5–3.0.
  - So the taker offset (≈ ±1 in round 1) and the maker half-width both scale up about ×2–2.5 in the volatile regime. The
    rule looks like round(S ± k·1) with k regime-dependent; k ≈ 2.5 is a guess.
  - In calm, the round-1 check (prior maker bids ≤ p − 1 for a buy) holds with 1 exception in 22.7k. About 5 % of other-side
    maker quotes sit below p − 1, which needs a look by the S agent.
- **Size.** P(v = 1…6) matches the rounded lognormal to 4 decimals. Log-size skew 0.004, kurtosis −0.03. The tail matches
  too: 45 takers over 5000 against 44 expected. Size is the same by side, regime and burst. There is no second taker
  type in size.

### 3. Makers (`makers.py`)
- **Size tail.** The rounded lognormal(2.657, 1.803) fits out to the extreme tail. Above 10k: 143 observed, 149 expected.
  Above 40k: 7 observed, 5.7 expected. Quantile 0.99999: 32.5k observed, 31.1k model. The 72k maximum is consistent with
  1.07M lognormal draws (expected max ≈ 98k). **No whale orders.**
- **Side.** P(buy) is 0.497–0.502 in every decile of book imbalance and of the 1 s and 10 s trend. Size and side are the
  same in the drought and after the jump, though the 3.0–3.5 s window shows buy share 0.480 ± 0.010.
- **Rate.** The same in both regimes (180.07 vs 179.83). It is cut to 0.30× only in the drought.
- **Placement.** In the volatile regime, makers quote much wider, and the touch-to-opposite distribution at insert moves
  from 1–2 ticks to 3–4.
- **Lifetimes.** Shorter in the volatile regime (median 0.21 s vs 0.31 s) and much shorter in the drought (median 0.14 s),
  so the book drains to about 1/8 depth before the jump. Cancel rows continue at about 180 /s for the first 0.5 s of the
  drought.

### 4. Seasonality, breaks, misc (`misc.py`)
- Order ids are consecutive from 1000001 with no gaps. Timestamps are strictly increasing and all distinct, with a uniform
  µs last digit.
- Per 1000 s block, maker rate is 179.2–181.2, calm taker rate per side 3.96–4.08 and volatile 5.83–6.39. Volatile share
  varies from 2 % to 30 % by block, which is random given dwell ≈ 250/44 s.
- The periodograms show no periodic line. The maker peak sits at 12× median power, about what the largest of 3000 noise
  bins would show.
- No other structural break. The large price trend (−480 then +230) is volatile-regime episodes with c ×5 (3816–4045:
  −270 in 229 s), plus shocks −88, −51, +114, +100.

## What a market maker can exploit
1. **Shock warning: 3 s, every time.** Maker inserts drop from about 90 to about 27 per side per second. A 0.5 s window
   with fewer than 50 maker inserts (normally 89 ± 9.4) detects it with no false positives in this tape.
   - Exactly 3.0 s later, S jumps by N(0, ≈43) ticks, against which any resting quote is pure adverse selection.
   - Pull everything before onset + 3.0 s and stay out until the new level is set.
   - Then provide liquidity into the symmetric taker burst (≈40 /s for 1–2 s) at the new level. Spreads are wide and the
     book is thin there.
   - Shock times are also roughly schedulable: none within 147 s of the previous one.
2. **Volatile regime, detectable within about 1 s** from spread > 1.5 or maker-envelope width > 2.5.
   - Impact per lot is ×5 and takers cross ≈2.5 ticks through S. Quote wider (≥ 2–3 ticks from S), cut size, or step
     aside.
   - In calm, round-1 tactics apply unchanged: same taker rate, offset and size, c ≈ 0.026–0.03.
3. There is no directional alpha in taker flow: sides are coin flips, with no meta-orders and no momentum. The only
   predictive signal is each taker's own √q impact.

## Open questions
- What is the exact rule that scales with regime? Taker offset, maker D and c all grow. Is it one volatility multiplier
  σ_t (≈ ×2.5 on offsets, ×5 on c)? To be fitted against the exact S from s_dynamics.
- Is there residual calm diffusion beyond c·√filled? Calm R² is 0.80 against 0.95 in round 1.
- Jump-size law (normal vs Laplace) and the gap law (U(150, 300) vs 150 + Exp): n = 27 can't separate them.
- Does a shock raise the chance of entering the volatile regime? 3 of 27 shocks are followed within 15 s, against 1.6
  expected, which is not significant.
- What is the drought cancel hazard and its exact shape? A higher α in the drought? The round-1 mv-based model is to be
  refit with the correct S.
