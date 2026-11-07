# Round 2: dynamics of the hidden fair price S

Tape `data/round2.csv` (6000 s, 1,123,282 inserts). Scripts in this folder are run with
`uv run --with pyarrow --with numba python marek/research/round2/s_dynamics/<script>.py`. The extra packages
are needed because the project venv has no numba or pyarrow; nothing in `pyproject.toml` was changed. Intermediate
results go to `_cache/` (git-ignored, about 660 MB).

**Output for other analyses:** [s_path.parquet](s_path.parquet) has one row per insert (1,123,282 rows).
- Its time `t_us` is the tape's µs time.
- `s_hat` is the smoothed posterior mean of S just before that insert. A taker's row is S *before* its own impact.
- `s_lo` and `s_hi` bound the 99 % credible interval.
- Extra columns: `s_sd`, the causal filtered `s_filt` and `s_filt_sd` (usable online), and the `vol` and `shock` labels used.

The regime switch times are in [vol_segments.csv](vol_segments.csv).

## Summary: the identified model

Round 2 is the round-1 generator with the same skeleton: makers quote around S, takers price IOCs off S, and filled
taker volume moves S. Three things change:
- a two-state volatility regime that rescales every price parameter at once;
- a new impact law with per-trade noise;
- 27 scheduled exogenous shock jumps.

| Part | Calm (85 % of time) | Volatile (15 %) | Confidence |
| --- | --- | --- | --- |
| Impact per filled taker | **ΔS = sign · c · filled^e + ξ** | same form | high |
| c | **0.0142** | **0.0741** (×5.2) | ±3 % |
| e | **0.625** (0.60–0.65) | **0.61** | high that e ≠ 0.5 (ΔNLL ≈ 155 at the same data; round-1 check below) |
| ξ, the per-filled-taker noise | N(0, sd **0.124**), independent of size and sign | N(0, sd **0.28**) | high in calm. In volatile it trades off against the jump term below. |
| Unfilled IOC | no impact, no noise | no impact | high |
| Diffusion between trades | **≈ 0**: σ ≤ 0.13 /√s, not distinguishable from 0 | σ ≈ 0.08 | high that it is small |
| Other exogenous jumps | none | Laplace(scale 0.84) jumps at ≈ 0.23 /s (weak, possibly misfit) | low |
| Maker offset D (bid = min(⌊S−D⌋, ask−1), ask = max(⌈S+D⌉, bid+1)) | **0.050 + Exp(0.454)**, exactly round 1 | **0.70 + Exp(0.90)** | high |
| Taker limit, regular | round(S ± **1**), uniform band [0.5, 1.5] | round(S ± **2.55**), band [2.05, 3.05] | high (2.5 vs 2.55 not resolved) |
| Taker limit, aggressive tail | ≈ 2–5 % of takers, extra offset ≈ 0.5 + Exp(≈ 3) ticks | ≈ 4–8 %, extra offset ≈ 0.5 + Exp(≈ 6) | medium (the shape is only roughly exponential) |
| Regime dwell (exact switch times) | mean 247 s (5.9–1026 s, 20 gaps) | mean 42.8 s (7.2–217 s, 21 episodes) | high |
| Shock jumps (27, from the flow agent's schedule) | S jumps by J ~ sd 42.9 (mean \|J\| 30) at drought onset + 3.0 s | same | high |

The volatile regime is not jump noise. The "5 jumps/s of 1–7 ticks" that a regime-agnostic smoother sees there are the
taker impacts with c ×5 and a 2.5× noise. Takers also sit 2.5 ticks off S, which a round-1 observation model misreads
as S jumping.

**Validation (final model, `validate.py v3`, [validation_v3.json](validation_v3.json)):**
- **Calm.** 0 of 908,240 maker inserts violate bid ≤ S − 0.05 or ask ≥ S + 0.05 at the point estimate. The smallest gap
  is 0.0501.
  - Takers: 94.7 % sit inside the round(S ± 1) band and 0 are on the passive side. The remaining 4.8 % are the
    aggressive tail, which the model includes.
  - The posterior sd of S has a median of 0.057 tick, and the 99 % interval has a median width of 0.25 tick.
- **Volatile.** 18 of 158,381 maker inserts fall below the regime's d0 = 0.70, and only 3 are inconsistent with the
  99 % interval. All 18 lie within 0.4 s of a regime switch, so they reflect the boundary timing, not the rule.
  - Takers: 90.9 % sit inside the 2.55 band and 4 are on the passive side.
  - The posterior sd has a median of 0.10 tick.
- **Shock windows** ([onset + 2.7, onset + 3.6] s): 0 maker violations. Taker fit is poor only for the few takers that
  meet the jump.
- **Method check on round 1** (`r1_check.py`, where the truth is c = 0.026, e = 0.5 and no noise): the same HMM
  recovers e = 0.5 and c = 0.0258. It rejects e = 0.6 (ΔNLL +262) and rejects per-taker noise of 0.016 (ΔNLL +1370).
  So the round-2 e ≈ 0.62 and ξ ≈ 0.12 are not artefacts of the method. The method does report a spurious
  σ ≈ 0.1 /√s on round 1 (ΔNLL −7), so the calm σ ≈ 0.13 in round 2 (ΔNLL −6) should be read as **zero diffusion**.

## How S was reconstructed (`prep.py`, `hmm.py`, `smooth_seg.py`, `full_v2.py`)

The filter is an exact grid HMM, with a forward filter and a backward smoother, over every insert:
- **Grid.** Step 0.05 tick, ±12 ticks around a moving centre. The centre is a rolling median of maker prices, so
  100-tick shocks stay inside the window.
- **Transition between consecutive inserts.** The previous taker's impact c·sign·filled^e is a deterministic shift.
  Gaussian noise follows: σ²·dt plus the per-filled-taker variance ξ². Then come a uniform "reset" jump at rate λr (set to
  30 /s inside shock windows) and a Laplace jump at rate λj.
- **Observation model, makers.** The exact round-1 likelihood for makers: P(⌊S−D⌋ = p). For an insert clipped at the
  opposite touch, it is the censored P(⌊S−D⌋ ≥ p).
- **Observation model, takers.** A band of probability 1 − ε at round(S ± off), plus an aggressive tail ε·Exp(τ) beyond
  the band.
- **Speed.** It runs in about 20 s on the whole tape with numba. Parameters were fitted by MLE (`fit_seg.py`,
  `grid_calm*.py`) on calm segments 700–840, 1100–1280 and 2780–3000 s, and on volatile segments 2031–2140, 3640–3730
  and 5620–5715 s.
- **Progression.** The total log-likelihood rose from −1,045,100 for the round-1-style model (`v1`) to −861,053 with
  regime observation parameters (`v2`), then to −851,230 for the final model (`v3`).

## Evidence by question

### 1. Do the 0.05 shift and the taker offset still hold?
- **Calm makers.** The interval-censored nonparametric EM for D (`dfit.py`) gives quantiles 0.1 %/50 %/99 % =
  0.10/0.40/2.15.
  - The shifted-exponential MLE is d0 = 0.0500, θ = 0.4536. A shifted gamma has shape 1.04, so the exponential is right.
  - This is identical to round 1.
- **Volatile makers.** The shifted-exponential MLE is d0 = 0.713, θ = 0.894, mean 1.61; a gamma has shape 0.91.
  - No maker quote comes within 0.70 of S in a volatile period, against 74 % of calm quotes.
  - This is the sharpest regime signature of all. The relation to the calm law (for example 2·D + 0.6) is not clean.
- **Calm takers.** The band is the round-1 round(S ± 1).
- **Volatile takers.** The band is round(S ± 2.55). Both volatile segments fit off = 2.55 independently, against
  2.5 ± 0.05 at grid resolution; whether the true value is 2.5 is unresolved.
- **The aggressive tail is new.**
  - Its extra offset has a uniform fractional part, so it is continuous, not a whole number of ticks.
  - It has no link to size, sign, time of day or the coming S move (`an3.py`, `tail.py`).
  - Its share rises from about 5 % in calm to about 10 % in volatile.

### 2. What moves S?
- **Impact law.** The window regression `calm_reg.py` and the HMM grid `grid_calm2.py` agree.
  - With σ and ξ fixed, the best c at each exponent gives NLL 67706 at e = 0.5, 67602 at 0.55, 67550 at 0.60, 67549 at
    0.65 and 67594 at 0.70.
  - The nonparametric impact curve (`impact_np.py`, smoother with no impact prior) is flat when divided by filled^0.6:
    c ≈ 0.0134–0.0142 calm and ≈ 0.068 volatile for filled from 8 to 1760.
  - It is not flat when divided by √filled: the ratio rises from 0.017 to 0.030.
  - Ordered size is worse than filled size (R² 0.76 vs 0.87 in 2 s windows). Unfilled IOCs move S by −0.0001 ± 0.0001.
- **Noise ξ per filled taker.** Compared on the same data (`grid_jvm.py`, NLL):

  | Noise structure | NLL |
  | --- | --- |
  | ξ² = 0.016 per filled taker | 67546 |
  | variance per maker insert only | 68212 |
  | variance ∝ filled | 67866 |
  | Brownian σ only, no ξ (fit A) | +450 worse |

  - Unfilled takers carry no noise (fit F, jvn → 0).
  - Window residual variance grows linearly with the number of filled takers, about 0.018 per taker, with an intercept of
    about 0 (`calm_reg2.py`).
  - ξ is symmetric with mean ≈ 0. It does not depend on fill prices or the number of levels swept (`impact_feat.py`).
- **No continuous motion.** Between trades S does not diffuse (see the round-1 check above). S moves only at filled
  taker events and at the shocks.
- **No momentum or mean reversion** (`memory.py`), using X = S − impact with shocks removed:
  - The variance ratio is 0.90–1.03 for horizons from 1 to 300 s.
  - The calm 1 s drift has t = −0.5 and the volatile 1 s drift t = +0.9.
  - The taker side stays a coin flip (flow report).
  - Per-episode net moves in volatile episodes are consistent with no drift: mean z² = 1.28 over 24 episodes.
  - The big trends, such as −270 in 3816–4033 s, are c ×5 impact random walks plus the −88 shock.
- **Shocks** (`shocks.py`, schedule from `../flow/shock_events.csv`):
  - The jump becomes visible in the first maker insert 2.96–3.20 s after the drought onset; the median is 3.02 s.
  - Sizes run from −87 to +113, with sd 42.9 and kurtosis 1.2.
  - S does not move during the drought beyond the normal impact: median \|ΔS\| is 1.5 over 2.9 s.
  - The jump is not predicted (corr(J, prior 10 s) = 0.28, n = 27).

### 3. Regimes
- **Two states, switching discretely.** Switch times come from a change-point search on the per-insert
  log-likelihood ratio of volatile against calm observation laws, given S (`switch.py`).
  - Each switch is decisive: the log-likelihood gain is about 1500–2400 within ±8 s.
  - The switch is located to within one or two inserts, about 10 ms.
  - The maker D law, the taker offset and c all change at the same instant.
  - The flow agent's taker-count HMM and envelope-width segmentation agree to ±0.5 s.
  - 21 volatile episodes cover 14.98 % of the time.
- **Taker rate is coupled to the regime.** It is 8.2 /s in calm and 12.7 /s in volatile (both sides; flow report:
  4.0 and 6.0 per side). Maker rate does not change (178.6 vs 176.8).
- **No continuous GARCH-like component.** Inside a regime, the calm variance ratio is flat and the 1 s residual
  kurtosis is about 0. The shocks are on their own schedule and are independent of the regime.
- **Detectable in real time from public data.**
  - A new maker insert closer than 0.7 tick to S cannot occur in the volatile regime. Using mid as a proxy for S, the
    insert distribution flips within tens of ms.
  - Flow agent: 1 s spread > 1.5 classifies 96.7 % of seconds correctly.
  - A causal CUSUM (threshold 10) on the filtered S log-likelihood ratio is a rough prototype, run on the v2 path with
    preliminary labels. It fires within 0.16 s of most onsets (median about 0.03 s); 2 onsets were late (6 s and 16 s)
    and a few alarms came early, before the labelled onset. Its false-alarm rate is high (658 in calm), so a production
    detector needs a better tail model for the taker likelihood.

## Still uncertain
- **The exact form of the volatile scaling.** The ratios are c ×5.2, taker offset ×2.55, ξ ×2.2 and D: 0.05 + Exp(0.45)
  → 0.70 + Exp(0.90). No single multiplier reproduces all of them. Possibly the generator draws separate per-regime
  constants.
- **Volatile-regime residual jumps** (Laplace 0.84 at 0.23 /s). They may be a misfit of the aggressive-taker tail or of
  the regime edges, not a real component.
- **e = 0.60–0.65 is resolved, the exact value is not.** Neither is whether the law is c·filled^e or c·√filled plus a
  size-dependent extra term. Calm and volatile both prefer e ≈ 0.61–0.63.
- **The shape of the aggressive taker tail.** It is roughly 0.5 + Exp, but the KS test rejects a pure exponential.
- **The shock jump law** (n = 27): normal and Laplace cannot be separated. Sd 43.

## Files
- **Core:** `common.py`, `prep.py` (per-insert arrays), `hmm.py` (numba filter/smoother), `smooth_seg.py`,
  `fit_seg.py` (MLE), `full_v2.py` (whole-tape smoother: `full_v2.py v3 params_v3.json`), `export_path.py`.
- **Parameter sets:**
  - [params_v3.json](params_v3.json): the final model.
  - `params_v2.json`: intermediate.
  - `params_np.json`: no impact prior, used for the nonparametric impact curve.
- **Analyses:** `validate.py`, `diag.py`, `dfit.py`, `impact_np.py`, `impact_feat.py`, `calm_reg*.py`,
  `calm_diff.py`, `grid_*.py`, `r1_check.py`, `switch.py`, `shocks.py`, `tail.py`, `memory.py`, `vol_jumps.py`.
- **Exploratory, on the v1 path:** `an1–an5.py`, `jumps.py`, `episodes.py`, `regimes.py`, `rates.py`, `peek.py`
  (`peek.py TAG t0 t1` prints events with S).
