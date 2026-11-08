# Round 4: dynamics of the hidden fair price S

Tape `data/round4.csv` (6000 s, 1,228,523 inserts). Scripts in this folder run with
`uv run --with pyarrow --with numba python <script>.py` (from this folder). The core filter is a copy of round 3's
(`common.py`, `prep.py`, `hmm.py`, `smooth_seg.py`, `full_v2.py`, `fit_seg.py`, `switch3.py`, `viterbi_seg.py`,
`validate.py`, `shocks.py`, `prof_full.py`), retargeted to round 4. Intermediate results go to `_cache/` (git-ignored).

**Output for other analyses:**
- [s_path.parquet](s_path.parquet): one row per insert (1,228,523 rows), round-3 columns: `t_us`, `s_hat` (smoothed
  posterior mean of S just before the insert; a taker's row is S *before* its own kick), `s_lo`/`s_hi` (99 % interval),
  `s_sd`, the causal `s_filt`/`s_filt_sd`, and the `vol`/`shock` labels. Final version = run `v3` (`params_final.json`,
  run with `VELFILE=_cache/kfd_v2b_D_state.npz`). Versions published on the board: v0 (round-3 params), v1 (v2b), final (v3).
- [vol_segments.csv](vol_segments.csv): exact regime switch times (32 volatile episodes).
- [model_params.json](model_params.json): the identified model.
- `_cache/ladder_member.npy` (ladder chains, lid/rung) and `_cache/react.npy` (reactive 15-lot makers) used as exclusions.

## Summary: the identified model

Round 4 keeps round 3's S law (decaying impact of filled takers, two regimes, scheduled shocks) and adds **one new
component: S has a velocity.** S drifts continuously between trades with a random, short-lived velocity V. V is large in
volatile (sd ≈ 3 tick/s, persistence ≈ 3 s) and small in calm (sd ≈ 0.4 tick/s). It is exogenous: fills do not feed it.
Nothing else about S changed beyond noise in the estimates.

The super-diffusion the lead found is this velocity plus a feedback through the takers (flow's side law: takers buy
when S is above the mid). V pushes S away from a lagging book, takers then trade on V's side, and their kicks push S
further the same way. The taker sign persistence is the side law, not a hidden direction state (flow).

| Part | Calm (70 % of time) | Volatile (30 %) | Confidence |
| --- | --- | --- | --- |
| State | S = F + Y + ∫V dt | same | high |
| Kick per filled taker | k = sign · c · filled^e + ξ | same | high |
| c | **0.0145** (r3 0.0168) | **0.048** (r3 0.062) | ±5 %, ±8 % (trades off with V) |
| e | **0.62** | **0.646** | high |
| ξ sd per fill | ≈ **0.10** (r3 0.13) | ≈ **0.22** (r3 0.26) | medium (V absorbs part of it) |
| Kick decay K(τ) = p + (1−p)e^(−τ/T) | **p 0.56, T 14 s** | **p 0.61, T 17–18 s** | as r3; ±0.05 in p, ±3 s in T |
| **Velocity V (new)** | sd ≈ **0.4 tick/s**, persistence ≈ 1–1.5 s; plus trend episodes flagged by the 15-lot agent (makers: ~0.7 tick/s for ~8 s) | sd ≈ **3.1 tick/s**, persistence ≈ **3 s** (KF 1.6–1.9 s), Gaussian | vol: medium-high; calm: low |
| V coupling to fills | none (κ calm −0.03, vol +0.02, ΔNLL 4.6) | none | high |
| V at regime switches | collapses to calm level at vol end; turns on at vol start. V leads the quote/kick switch by ≈ 0.75 s at both ends | | medium |
| Mean drift | 0 (net +10 ticks over 4061 s) | −0.37 tick/s (net −632 over 1730 s, t ≈ −2.4) | low |
| Maker offset D (non-ladder, non-agent) | **0.032 + Exp(0.469)** | **0.676 + Exp(0.995)** | high (= r3) |
| Taker limit | round(S ± 1), tail 2.3 % × Exp(3.1) | round(S ± 2.55), tail 6 % × Exp(6.6) on clean segments | high |
| Regime | 2-state; calm mean dwell 134.6 s | 32 episodes, 29.8 % of time, mean 55.8 s (5.7–138.5 s) | high |
| Shocks (28) | S += J at drought onset + **3.02 s**; **J ~ N(0, 63)** (mean \|J\| 48; r3 sd 48); no reversion | same | high (sd ±9) |

For a simulator: per small step dt, V follows an OU process, dV = −V/τ dt + σ√(2/τ) dW with stationary sd σ
(vol σ 3.1, τ 3 s; calm σ 0.4, τ ≈ 1 s), and S += V dt. Set σ from the regime at each step. I did not separate OU from
"piecewise-constant V with random resets": the reset model fits better at short range, the OU fits the long-range
variance (see question 2). Kicks and shocks as round 3.

## How S was reconstructed

The round-3 grid HMM (step 0.05 tick, ±12 ticks around a rolling maker median) with these changes:
1. **Exclusions** (observation likelihood 1): ladder chains, i.e. maker inserts linked by exact 1999–2001 µs same-side
   gaps (21.8k chains, 66.0k inserts; `excl.py`), and the reactive 15-lot maker (maker 1 µs after an IOC, or a 15-lot
   ≤ 100 µs after an IOC: 395 inserts). Makers later published the full agent label (`makers/cache/order_class.parquet`,
   cls 2, 755 orders); the ~350 re-posts I miss are absorbed by the 2.4 % outlier term.
2. **Velocity plug-in**: the per-step shift is the kick path (with decay) plus V(t)·dt, where V is the Kalman-smoothed
   velocity from `kfd.py` (state F, Y, V on the previous S path, observed at every IOC row). I iterated
   path → KF → path three times (v1a → v2a → v2b → v3). A small residual diffusion stays in the filter: calm 0.15,
   vol 0.25 tick/√s (profile optimum).
3. Observation laws refitted on segments (`fit_seg.py`; calm 2560–2680, 4140–4250, 4880–5000 s; vol 1950–2075,
   3641–3777, 5146–5240 s). They equal round 3's.
4. Regimes: a per-second Poisson Viterbi on multi-rung ladder bursts and takers (`seg_init.py`) gives the start; then
   round 3's per-insert llr Viterbi (`switch3.py`, `viterbi_seg.py`, penalty 100). Re-running it on the final path
   changes no switch by more than 0.38 s (median 0).

Total HMM log-likelihood: −1,098,290 with round-3 params and ladder exclusion (S v0); −1,087,856 with only a vol
diffusion of 1.1/√s; −1,072,705 (S v1); **−1,071,071 (final)**. The plug-in uses a two-sided velocity estimate, so these
gains overstate the evidence. The model comparisons below use the Kalman filter and the velocity HMM instead.

## Evidence by question

### 1. Does S have its own drift, or is the super-diffusion all impact from persistent flow?
S has its own drift. Four results:
- **Likelihood (Kalman filter on the path, `kfd.py`).** Adding a velocity state improves the NLL by about 9,800 nats
  (S v1a: −7,831 → −17,658). A plain diffusion instead of V gains only ~1,600 (S v0 path: impact-only 12,450 → diffusion
  10,842 → velocity 7,950).
- **No feedback from fills into V.** Letting each kick add κ·k to V gives κ calm −0.03, vol +0.024 and ΔNLL 4.6 for
  2 parameters. Novelty found the same model-free: past taker imbalance adds nothing once the past residual move is in.
  Flow found that given S − mid the taker signs are independent. So V is exogenous. Takers follow S; S does not follow
  takers.
- **Moves between fills.** On the round-3 model (S v0), the HMM wants a vol diffusion of 1.1/√s: +10,300 nats, where
  round 3 had none. Novelty: in 0.5 s windows with no fills, mean |ΔS| is 1.11 in vol (r3 0.16) and 0.10 in calm
  (r3 0.017), and the move continues the past 2 s.
- **Impact kernel.** Without V the vol kernel looks almost permanent (p 0.75–0.83, T 29–39 s), because the drift
  leaks into it. With V it returns to round 3 (p 0.61, T 17 s).

**Variance decomposition** (`decomp.py`, final path, 1 s grid, shocks removed). S = drift integral + kick path:

| h | var S | var drift | var impact | 2·cov | VR(S) |
| --- | --- | --- | --- | --- | --- |
| 1 s | 21 | 2.5 | 14.6 | 3.6 | 0.97 |
| 10 s | 554 | 110 | 260 | 183 | 2.58 |
| 60 s | 2,297 | 755 | 804 | 731 | 1.79 |
| 300 s | 12,110 | 5,889 | 2,151 | 4,027 | 1.88 |

At 300 s the drift itself gives 49 % of the variance, impact alone 18 %, and the covariance 33 %. The covariance is
the feedback: V moves S off the mid, the side law sends takers V's way, and their kicks add to the move (corr of 1 s
drift with impact in the same or next second 0.29–0.32). The decaying kicks alone would be sub-diffusive, as in round 3
(the impact VR falls to 0.32 at 600 s).

### 2. The velocity law
- **Volatile.** On the 1 s grid, residual increments (S minus kicks) have ACF 0.74 / 0.54 / 0.39 / 0.18 / 0.08 / 0.01
  at lags 1 / 2 / 3 / 5 / 7 / 10 s, an exponential with τ ≈ 3 s, and nothing beyond 10 s (`vlong.py`). Local velocity
  on 1–4 s windows is Gaussian (excess kurtosis 0.1–0.35, `velacf.py`) with sd 2.2–2.8 tick/s, noise included.
  Estimates of σ and τ:
  - KF OU: σ 3.0–3.1, τ 1.6–2.1 s.
  - Velocity grid HMM (`vhmm.py`, 0.25 s bins, fill noise ∝ fill count): OU σ 2.93, τ 3.1 s.
  - Long-horizon drift variance: 5,889 at 300 s, which needs τ ≈ 3 s with σ ≈ 3.1.
- **OU or resets?** The velocity HMM ranks the models by NLL: OU −2,471, piecewise-constant with Poisson resets −3,059
  (mean life 1.2 s vol, 1.0 s calm), OU plus resets −4,041. But the reset fit's 1.2 s life cannot produce the 3 s ACF.
  The OU-plus-resets fit degenerates (calm τ → 332 s, with resets every 28 s). So V is persistent with occasional larger
  changes, and the exact law is open. On/off episodes in calm fit worse (−2,724).
- **Calm.** The velocity is small and hard to separate from ξ:
  - KF: τ 0.26 s, sd 0.7. This is diffusion-like and absorbs ξ, whose estimate falls from 0.017 to 0.002 over the
    iterations.
  - vhmm: sd 0.36–0.44 tick/s, τ 1–1.5 s, ξ variance 0.009–0.010 per fill.
  - Raw-data HMM without a plug-in: noise at fills (r3 ξ, no diffusion) beats a continuous diffusion (σ 0.3, small ξ)
    by about 3,000 nats. So most calm randomness still arrives at fills, and the calm V is weak.
  - The 15-lot agent (makers) marks calm trend episodes (31 episodes, median 8 s, ~0.7 tick/s, side-signed move +8.4 to
    the episode end). These are above the calm V's sd, so calm may hold rare stronger trends on top of a weak V.
  - Novelty found a box-like calm ACF of about 4 s.
- **Not on a clock.** Pairs of 0.1 s increments correlate the same inside a 1 s (or 0.5, 0.7, 1.3, 2 s) period as
  across its boundary, and |ΔV| is flat over the phase of the second (`clock.py`).
- **Regime switches** (`swvel.py`). V collapses at the vol end: 4 s mean slopes before and after are uncorrelated.
  Sign-aligned V around switches:
  - Calm → vol: V rises from −0.75 s (0.7, 1.5, 1.3, then 2.8 at 0 s; tick/s).
  - Vol → calm: V falls from −1.0 s (2.1, 1.4, 0.8, 0.6, then 0.2 after the switch).
  - Kick sizes on the no-prior path switch exactly at the detected time (calm 0.012 before, vol 0.057 after;
    `preswitch.py`), and so do the observation laws (support of the change point ≈ 40 ms).
  - So V's regime seems to lead the quote/kick regime by ≈ 0.75 s at both ends. Medium confidence: it rests on
    30 switches per direction.
- **Mean drift.** In vol the net residual drift is −632 ticks over 1,730 s (−0.37 tick/s, t ≈ −2.4 with the
  autocorrelation allowed for), and 21 of 31 episodes are negative. Calm: +10. It may be a negative μ in volatile or
  chance. Low confidence. Over the whole tape S went 10,000 → 9,556 (min 9,183).

### 3. Impact law
With V in the Kalman filter (S v2b path, 58k IOC observations), the parameters are stable across path iterations:
- calm: c 0.0145–0.0153, e 0.619–0.622, p 0.54–0.57, T 13.9–14.4 s;
- vol: c 0.048–0.050, e 0.643–0.646, p 0.58–0.61, T 16–18 s.

The exponent from the no-prior smoother (`impact_np.py`) is again flat in filled^0.6 (best e 0.6 in both regimes).
Calm c is 13 % below round 3 and vol c 23 % below. This part of the impact trades off against V and the side-law
feedback, so treat c as ±10 %. The decay clock is assumed to be clock time, as in round 3 (not retested). Unfilled
IOCs: no kick (sign·ΔS −0.001 / −0.002).

### 4. Regimes
32 volatile episodes, 29.8 % of the time; mean dwell vol 55.8 s, calm 134.6 s (r3 28 %, 60 s / 157 s). Flow: calm
dwell exponential; vol dwell not (gamma k ≈ 2.7, low n). The switches are decisive: the evidence within ±5 s is
1,300–3,500 nats each.

An early version had 12 of 32 volatile starts within 0.05 s of an integer second. On the exact change points of the
final path it is 4 of 32 against 1.5 expected (ends 4 against 1.9). Treat it as an artefact.

### 5. Shocks (`shocks.py` on S v1; schedule from `flow/shock_events.csv`)
- 28 jumps, sd 63.2, mean \|J\| 48.3, mean +2.0, excess kurtosis 0.13. The normal fits (KS p 0.99, NLL 155.3; Laplace
  156.0). Only 2 of 28 have \|J\| < 3, so there is no evidence of a no-jump class.
- The jump comes 2.98–3.06 s after drought onset (median 3.02).
- No reversion: the slope of the post-jump move on J is 0.01 / 0.09 / −0.07 / −0.20 at 10 / 30 / 60 / 120 s.
- No pre-onset tell: corr(J, prior 10 s) = −0.20.
- The S move during the drought has sd 5.8 (r3 2.8). This is V running through the drought.

## Validation (`validate.py v3`)
- **Calm:** 119 of 768,845 non-ladder maker inserts fall below d0 (0.015 %), and 8 are inconsistent with the 99 %
  interval. Takers: 94.1 % in round(S ± 1), 5.3 % in the tail, 1 on the passive side. Posterior sd median 0.062.
- **Volatile:** 197 of 330,484 fall below d0, 69 inconsistent with the 99 % interval. Takers: 87.0 % in band,
  11.8 % in the tail. Flow attributes the larger tail to tail takers leaning on S − mid; on clean segments the tail is
  6 %. Posterior sd median 0.11.
- **Shock windows:** as in round 3; the takers that meet the jump fit poorly (62 % in band).
- **Model checks:** the impact law is unchanged when V is added. κ (fill → V) ≈ 0. Regimes are stable between S
  versions. The tests that found V (moves between fills, ACF, KF) all come out flat on round 3's S (novelty: r3 residual
  ACF all < 0.09). I ran no synthetic-tape check of the velocity estimator.

## Still open
- **The velocity law.** OU against piecewise-constant resets against a mixture. τ is 1.6–3.1 s depending on the
  method. The calm velocity cannot be separated from ξ, and how the 15-lot trend episodes relate to V is open. A 2-D
  (S, V) filter or a closed-loop simulation (V + side law + kicks) would settle it.
- **The ≈ 0.75 s lead of V over the regime switch.** Unexplained; it could be a lag in the agents' regime.
- **A negative mean drift in volatile** (t ≈ −2.4).
- **ξ in calm.** About 0.10 per fill, but it trades off against V.
- **The S path is two-sided.** It uses a smoothed V plug-in. `s_filt` is the causal filter given that plug-in, so it
  is not a clean real-time filter. For real time use the KF filtered state (`_cache/kfd_v2b_D_state.npz`, `filt`).

## Trading implications
- **S is partly predictable in volatile.** The causal KF velocity (`predict.py`) explains part of the next move:

  | Horizon | Vol R² | Vol β_v | Calm R² |
  | --- | --- | --- | --- |
  | 0.5 s | 0.22 | — | 0.03 |
  | 1 s | 0.29 | 1.7 | — |
  | 2 s | **0.33** | 3.0 | 0.03 |
  | 5 s | 0.24 | 5.0 | — |
  | 10 s | 0.14 | — | — |

  β_v saturates at about 5 s of V, longer than τ, because the takers' feedback amplifies the drift. In vol, a market
  maker should lean quotes toward S + V·(1–2 s), or at least not rest on the side V is moving toward. The strategy agent
  found that its existing position already harvests the trend (carry 637k of 775k in vol). Plain F-leans lost money
  there, but a larger vol filter diffusion (≈ V) gained +16k.
- **Toxic flow follows V.** Takers trade on the side S has moved toward relative to the mid (flow). In vol, being
  filled on the side of a fresh move is adverse: S is already past your quote, and V keeps going for about 3 s.
- **The transient part still reverts.** Y decays with T ≈ 14–18 s (β_Y −0.4 at 5 s, −0.9 at 10 s in vol). It is small
  at quoting scale, as in round 3.
- **Tells for V:**
  - The 15-lot agent's side in calm (makers: +5 ticks over the next 5 s after its first re-post).
  - TWAP side (novelty: pure impact, ~0.37 tick per child).
  - S − mid itself.
  - At vol onsets V is already running in the second before the quotes widen. A fast price move in calm with no fills
    is an early warning of a volatile episode.
- **Shocks:** J sd 63 (bigger than r3's 48); the 3.02 s drought warning and the stale-side rule carry over.

## Files
- **Core:** `common.py`, `prep.py`, `hmm.py`, `smooth_seg.py` (adds the reactive-maker exclusion and the `VELFILE`
  velocity plug-in), `full_v2.py`, `export_path.py`, `validate.py`, `excl.py` (exclusion masks).
- **Regimes:** `seg_init.py` (Poisson Viterbi on bursts and takers), `switch3.py`, `viterbi_seg.py`, `switch_frac.py`.
- **Drift:**
  - Kalman filter with V and RTS smoother: `kfd.py` (variants A, Ap1, AF, D, DK).
  - Velocity-law HMM: `vhmm.py` (OU / RESET / MIX / ONOFF).
  - Residual ACF and drift tests: `drift.py`, `velacf.py`, `vlong.py`, `clock.py`.
  - Velocity and kicks around regime switches: `swvel.py`, `preswitch.py`.
  - Variance decomposition and net moves: `decomp.py`, `netmove.py`.
  - Causal predictability: `predict.py`.
- **Impact:** `propagator.py`, `gfit.py`, `impact_np.py`, `kickres.py`, `prof_full.py`, `fit_seg.py`.
- **Shocks:** `shocks.py`.
- **Parameters:** `params_v0*.json` (round-3 params) → `params_v1a.json` (vol diffusion) → `params_v2a/b.json`
  (velocity plug-in) → `params_v3.json` = `params_final.json`; `params_np.json` is the no-impact-prior smoother.
