# Round 3: dynamics of the hidden fair price S

Tape `data/round3.csv` (6000 s, 1,234,596 inserts). Scripts in this folder run with
`uv run --with pyarrow --with numba python marek/research/round3/s_dynamics/<script>.py` (from this folder). Most are
copies of the round-2 scripts with the paths changed. Intermediate results go to `_cache/` (git-ignored).

**Output for other analyses:**
- [s_path.parquet](s_path.parquet): one row per insert (1,234,596 rows), the same columns as round 2. `t_us` is the
  tape's µs time. `s_hat` is the smoothed posterior mean of S just before that insert (a taker's row is S *before* its own
  kick). `s_lo` and `s_hi` bound the 99 % credible interval. Also `s_sd`, the causal filtered `s_filt` and `s_filt_sd`,
  and the `vol` and `shock` labels used. Final version = run `final` (`params_final.json`).
- [vol_segments.csv](vol_segments.csv): exact regime switch times (28 volatile episodes, `start`, `end` in s from tape
  start, plus the log-likelihood evidence).
- [shocks.csv](shocks.csv): the 27 shock jumps (time, lag after drought onset, size J, moves before and after).
- [model_params.json](model_params.json): the identified model. [params_final.json](params_final.json): the same in the
  HMM's parameter format.

## Summary: the identified model

Round 3 keeps the round-2 skeleton: makers quote around S, takers price IOCs off S, filled taker volume kicks S, a
two-state volatility regime rescales everything, and 27 scheduled shocks jump S. What changed for S:
- **The kick is now partly transient.** About half of each kick decays back with a time constant of about 15 s. Shock
  jumps do not decay.
- **A ladder maker** (found by novelty/flow/makers) quotes off the touch, not off S. Its orders must be left out of the S
  likelihood; otherwise they drag S (gaps to S go negative).
- Calm c is about 18 % higher than in round 2 and volatile c about 16 % lower. The volatile regime takes 28 % of the
  time (r2 15 %).

| Part | Calm (72 % of time) | Volatile (28 %) | Confidence |
| --- | --- | --- | --- |
| Kick per filled taker | **k = sign · c · filled^e + ξ**, S jumps by k at once | same form | high |
| c | **0.0168** (r2 0.0142) | **0.062** (r2 0.0741) | ±3 % calm, ±5 % vol |
| e | **0.616** (0.61–0.62) | **0.63** (0.628–0.644) | high that e ≈ 0.62, as in r2 |
| ξ | N(0, sd **0.130**) (r2 0.124) | N(0, sd **0.26**) (r2 0.28) | high |
| **Decay of the kick (new)** | K(τ) = p + (1−p)·e^(−τ/T), **p = 0.50** (0.45–0.55), **T = 16.7 s** (13–21) | **p = 0.55** (0.50–0.60), **T = 13.6 s** (11–17) | high that it decays; p and T ±10–20 %. Whether p and T really differ by regime is medium |
| Decay clock | clock time, not taker count | same | high (ΔNLL 807) |
| Does ξ decay too? | probably yes, with the same kernel | probably | medium (see below) |
| Unfilled IOC | no kick, no noise | same | high |
| Diffusion between trades | none (σ fit 0.08 /√s, ΔNLL < 1) | none (σ 0.01) | high |
| Other exogenous jumps | none | Laplace(0.92) at 0.18 /s improves NLL by 30 on 70k events | low (same weak signal as r2's 0.23 /s, 0.84) |
| Maker offset D (non-ladder) | **0.049 + Exp(0.476)** (r2 0.049 + Exp(0.450)) | **0.673 + Exp(1.00)** (r2 0.70 + Exp(0.91)) | high |
| Maker outliers | 2.6 % uniform: hidden one-rung ladder orders | 2.6 % | medium |
| Taker limit | round(S ± **1**), tail 2.45 % × Exp(3.4) | round(S ± **2.55**), tail 4.6 % × Exp(5.9) | high (= r2) |
| Regime | 2-state Markov; mean dwell **156.7 s** (r2 247) | mean dwell **60.3 s** (4.7–249 s, 28 episodes; r2 43) | high |
| Shock jumps (27) | J at drought onset + **3.03 s**; sd **48.1**, mean \|J\| 33.2; no reversion | same | high (law: normal vs Laplace unresolved) |

Momentum or mean reversion beyond the kick decay: none. Shocks are permanent. Taker signs stay coin flips (flow).

## How S was reconstructed

The filter is round 2's exact grid HMM (`hmm.py`, `smooth_seg.py`, `full_v2.py`) with three changes:
1. **Ladder orders are uninformative.** Inserts in the flow agent's exact-2000-µs chains (`flow/cache/ladder_member.npy`,
   69k inserts) get observation likelihood 1. Their gaps to S run from −1.5 to 9 ticks; base makers never go below d0.
2. **Outlier maker component.** A maker's likelihood is (1 − w)·P(round-1 D law) + w·P(D ~ U(−1, 9)), w ≈ 2.6 %.
   This absorbs the one-rung ladder orders that the 2 ms test cannot see.
3. **Transient kick.** The deterministic part of the kick decays: a running transient stock
   R = Σ (1−p)·c·sign·f^e·e^(−Δt/T) is computed over the whole tape. The shift between inserts i and i+1 is
   the kick of i minus the decay of R over that interval. ξ is kept permanent inside the filter (a 1-D state cannot
   carry a decaying random part). The synthetic check below shows this still recovers c, e, ξ, p and T.

Grid step 0.05 tick, ±12 ticks around a rolling median of maker prices. Shock windows [onset + 2.5, onset + 3.8] s get a
uniform reset rate of 20 /s. Regime labels come from `vol_segments.csv`.

Parameter fitting:
- Observation laws: segment MLE (`fit_seg.py`) on calm 1260–1350 and 1365–1525 s, and on volatile 700–745, 755–835,
  2330–2370 and 2385–2520 s.
- Dynamics: a Kalman filter on the smoothed path (`kf_fit.py`, `kf3.py`, state S = F + Y). Then one-at-a-time profiles
  of the full-tape forward log-likelihood (`prof_full.py`) for c, ξ, p and T.
- Total log-likelihood: −1,088,870 (r2 parameters + ladder exclusion), −1,078,132 (v4), **−1,077,515 (final)**.

## Evidence by question

### 1. Is impact still permanent?
No. Three independent checks agree, and the same checks are flat on round 2.
- **Propagator regression** (`propagator.py`, `gfit.py`; ΔS in 0.5 s bins on lagged signed filled^0.6, shocks
  masked). The cumulative response G(τ) is:

  | τ | 0 s | 5 s | 10 s | 20 s | 30 s | 60 s | 120 s |
  | --- | --- | --- | --- | --- | --- | --- | --- |
  | calm | .0188 | .0162 | .0142 | .0120 | .0111 | .0097 | .0096 |
  | vol | .0713 | .0620 | .0552 | .0480 | .0431 | .0395 | .0388 |

  The fit c·[p + (1−p)e^(−τ/T)] gives calm p 0.48, T 16 s and vol p 0.55, T 14.6 s. A power law degenerates to the
  exponential. Novelty found the same with an S proxy.
- **Response after single takers** (`decay.py`). r3 calm c_eff falls 0.0165 → 0.0091 from 0.2 s to 20 s. On round 2's
  S, the same code is flat (0.0151 → 0.0149).
- **Likelihood.** The full HMM prefers decay; in calm, p = 0.50 beats p = 1 (permanent). The Kalman filter with
  permanent impact is worse by ΔNLL ≈ 1500. Decay in clock time beats decay per taker event by ΔNLL 807.
- **Method checks.** The same profile prefers permanent impact on round 1 (p = 1 beats 0.5 by 84 nats) and on round 2
  (by 27 nats). On a synthetic r3 tape with known truth (`strategy/r3_synth.py` seed 1: calm c .0171, e .61,
  p .583, T 17.3; vol .075, .606, .55, 15), the pipeline returns:
  - Kalman filter: c .0172, e .609, p .582, T 18.0 (calm) and c .073, p .53, T 16.3 (vol).
  - Full-tape profile: ξ variance at its true values (.0146 / .076) and p = .583 best.
- **Shocks do not revert.** The slope of the move over the next 30/60/120 s on J is 0.02/0.06/0.06. So this is decay
  of the kick, not an OU pull of S toward a level. A regression with an interval-fixed anchor finds no pull at all.

### 2. Does ξ decay too?
Probably, with the same kernel, but the evidence is indirect.
- **Kalman filter: not usable.** On the real path it says ξ is permanent in calm (px → 1). On the synthetic tape,
  where ξ truly decays, it also returns px = 1. So that test cannot see ξ decay; it does not count against it.
- **Residual variogram** (`resid_var.py`, ±5 s around switches and big misfits removed). E = S minus the deterministic
  kick path. V(h)/h for h = 3 s → 120 s:
  - r3 calm: 0.148 → 0.054–0.062
  - synthetic with decaying ξ (truth p 0.58): 0.112 → 0.048
  - r2 (permanent): 0.141 → 0.157

  The r3 ratio of about 0.4 matches decaying ξ with p ≈ 0.6. Permanent ξ would leave the ratio at 1.
- **For a simulator:** decay the whole kick (c·f^e + ξ) with K(τ). That is what `strategy/r3_synth.py` does with
  `tr_xi=1`.

### 3. Exponent and size
- The nonparametric curve on a smoother with no impact prior (`impact_np.py`) is flat when divided by filled^0.6:
  calm 0.0136–0.0146, vol 0.058–0.064 across filled from 3 to 1300.
- Ordered size: not tested again; round 2 found filled better.
- Unfilled IOCs: sign·ΔS = −0.003 ± 0.002 (calm), +0.0007 ± 0.0015 (vol).
- Kick fills against ladder orders count like any fill (novelty).

### 4. Regimes
- **Still two states; no third.** Per-episode estimates are homogeneous (`episodes.py`, final path):
  - Taker offset median: 1.03 ± 0.03 over 29 calm stretches and 2.55 ± 0.04 over 28 volatile episodes.
  - Maker gap 1 % quantile: 0.37 ± 0.04 calm and 0.86 ± 0.01 volatile.
  - Taker rate: 7.6 ± 0.7 calm and 11.1 ± 1.1 volatile.
- **The lead's episodes are the volatile regime.** The "+15–20 % maker inserts, larger sizes" minutes (10–13, 38–41,
  63–68, 92–98) are volatile episodes 592–627, 682–840, 2274–2524, 3753–3885, 3937–3995, 5543–5658 and 5770–5935 s.
  The extra makers are the ladder, which runs 10× faster in volatile (flow, makers). S behaves there as in any volatile
  episode.
- **Switch times** (`switch3.py` + `viterbi_seg.py`): a 2-state Viterbi with switch penalty 100 on the per-insert
  log-likelihood ratio of the volatile vs calm observation laws, given S. Ladder inserts and the drought+jump windows
  are excluded.
  - Every switch is decisive: the evidence within ±5 s is mostly 1,700–3,700 nats (less where a drought is adjacent).
  - The times are stable to < 0.25 s between S v0 and v1. They agree with flow's taker/ladder Viterbi
    (median |Δ| 0.2–0.5 s).
  - A penalty of 30 adds only 1-s blips, mostly just before or after droughts.
- **Dwell:** 28 volatile episodes covering 28.2 % of the time. Volatile mean 60.3 s (4.7–249 s); calm mean 156.7 s
  (3.9–571 s). Flow: both exponential, exit rates 0.0065 and 0.0166 /s.

### 5. Shocks (`shocks.py` on the final path; schedule from `flow/shock_events.csv`)
- **Timing:** the jump shows in the first maker inserts 3.00–3.08 s after drought onset (median 3.03 s). The 4 cases
  with lags 2.6–3.8 s are jumps with \|J\| < 2, where the max-|ΔS| detector picks a taker kick instead.
- **Size:** −110 to +126, sd 48.1, mean \|J\| 33.2, kurtosis 1.5. MLE on r3: normal sd 47.4 (NLL 142.5), Laplace scale
  33.2 (140.3).
  - 6 of 27 jumps have \|J\| < 3, more than a normal (1.3) or Laplace (2.3) predicts. A "no jump" class of about 18 %
    fits best on r3 (NLL 138.3). On r2+r3 pooled the Laplace is best, so this is unresolved.
- **No drift in droughts:** sd 2.8 over the 3 s, which is just the impact.
- **Not predictable:** corr(J, prior 10 s) = −0.20 (n = 27).
- **No reversion** (see question 1).

## Validation (`validate.py final`, [validation_final.json](validation_final.json))
- **Calm.** 84 of 789,532 non-ladder maker inserts (0.011 %) fall below d0 = 0.049 at the point estimate, and 5 are
  inconsistent with the 99 % interval. 20 of them are within 2 s of a regime switch; the rest are the outlier class.
  - Takers: 94.7 % inside round(S ± 1), 4.8 % in the aggressive tail, 6 on the passive side.
  - Posterior sd median 0.062 tick; 99 % interval median width 0.30.
- **Volatile.** 188 of 315,203 (0.06 %) fall below d0 = 0.673, 83 inconsistent with the 99 % interval. The 0.1 %
  quantile of the gap is 0.73.
  - Takers: 90.7 % inside round(S ± 2.55), 8.4 % in the tail, 16 on the passive side.
  - Posterior sd median 0.107.
- **Shock windows:** taker fit is poor only for the takers that meet the jump (74 % in band), as in round 2.
- **Method checks:** round 1, round 2 and the synthetic r3 tape (question 1). The Kalman-filter ξ test fails on the
  synthetic tape and is not used.

## Still open
- **Whether ξ decays exactly like the deterministic kick** (question 2). The variogram says yes with p ≈ 0.6, but a
  direct likelihood test needs a 2-D filter (S, Y).
- **Whether p and T differ between regimes.** The Kalman filter prefers separate values: calm 0.50 / 17 s against
  vol 0.55 / 13.6 s (ΔNLL 15 for 2 parameters). Full-tape profiles are flat within ±0.05 in p and ±3 s in T. A single
  p ≈ 0.5, T ≈ 15 s is not excluded.
- **The exact c ratio.** Calm/vol c is 3.7 against round 2's 5.2. No single multiplier reproduces both rounds.
- **Volatile residual jumps** (0.18 /s, Laplace 0.92): possibly a misfit of the taker tail or the regime edges, as in
  round 2.
- **About 60 large S moves (> 0.2 tick) between takers, away from switches and shocks.** They are clustered (e.g.
  740.7 s, 3235–3237 s) and look like local misfits, not a jump process (0.01 /s).
- **Shock J law:** normal, Laplace, or a mixture with a point mass at 0.

## Trading implications
- **Filter.** Drop ladder orders (exact-2000-µs chains, plus a few-% outlier term) from any fair-value filter. This was
  the big fix in the strategy agent's tests (451k → 750k).
- **Transient stock.** Keep Y = Σ (1−p)·k·e^(−Δt/T). S drifts at −Y/T per second, so about half of any fill-driven
  move comes back with an e-folding time of about 15 s.
  - The size is small at quoting scale: typical calm |Y| is about 1 tick, so the drift is about 0.05 tick/s. The
    strategy agent found that skewing quotes by Y does not pay; including it in the filter gives about +5k.
  - It matters more for longer holds and after large one-sided sweeps in volatile, where c is ×3.7.
- **Shocks and regimes are as in round 2.**
  - Shocks: the jump comes 3.03 s after drought onset, sd 48, and it stays.
  - Regime detection: no volatile maker quotes within 0.67 of S; taker offset 2.55. The volatile regime is now 28 % of
    the time and also shows up as ladder bursts and 25–55-lot maker sizes.

## Files
- **Core:** `common.py`, `prep.py`, `hmm.py` (adds the outlier maker term), `smooth_seg.py` (ladder exclusion,
  transient kick), `fit_seg.py`, `full_v2.py` (`full_v2.py final params_final.json`), `export_path.py`, `validate.py`.
- **Regimes:** `seg_init.py` (first per-second segmentation), `switch3.py` (llr), `viterbi_seg.py` (exact switches),
  `episodes.py`.
- **Impact and decay:**
  - Response and kernel: `impact_np.py`, `decay.py`, `propagator.py`, `gfit.py`.
  - Kalman filters: `kf_fit.py`, `kf3.py`.
  - Full-tape likelihood profiles: `prof_full.py`.
  - Residual diagnostics: `resid_var.py`, `resid_acf.py`, `resid_count.py`, `resid_event.py`, `maker_jumps.py`,
    `kick_sd.py`, `xi_decay.py`, `mr_reg.py`, `memory.py`.
- **Shocks:** `shocks.py`; also `droughts.py` (early, superseded by flow's schedule).
- **Method checks:** `r1_check.py`, `r2_check.py`, `synth_prep.py` (synthetic tape inputs).
- **Observation laws:** `dfit.py`, `diag1.py`, `quickdiag.py`.
- **Parameter history:** `params_v1.json` → `params_v2.json` (published as S v0) → `params_v3.json`, `params_v4*.json`
  (S v1) → `params_final.json` (final S).
