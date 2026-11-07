# fv_mm: Bayesian fair-value market maker for round 2

[fv_mm.py](fv_mm.py) quotes and takes liquidity against a posterior over the generator's hidden price S. It does not use the visible book as its reference.

All numbers come from `fold_eval.py` on `data/splits_r2/wf` (12 folds × 370 s, each with a 60 s warmup), unless a row says otherwise.

## Results

| Strategy | r2 folds mean | Worst fold | Holdout `hold25` (1500 s) | Full round-2 tape | Synthetic r2 scenarios (7 × 1500 s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `marek/round1/robust_ladder` | +1.1k | −14.3k | −6.3k | | −4.7k |
| `peter/regime_mm` | +9.1k | +5.6k | | | |
| `marek/round2/r2_ladder` | +24.9k | +4.7k | +87.1k | | +106.8k |
| fv_mm v1 (single-regime filter) | +28.1k | +17.5k | | +437.7k | +129.3k |
| **fv_mm (defaults)** | **+36.7k** | **+19.1k** | **+117.3k** | **+565.4k** | **+148.6k** |

- fv_mm beats r2_ladder on every synthetic tape. The scenarios are base ×2, volatile-heavy, big impact, jump sd 80, no shocks, and extra diffusion. They come from `marek/research/round2/r2_synth.py`, which uses a √q impact law, so the filter's q^0.625 law is deliberately mis-specified there.
- On round-1 folds (`data/splits_r1/wf`) fv_mm makes +40.1k, sd 0.8k. s_mm, which knows the exact round-1 S, makes +45.6k.
- Runtime on the full round-2 tape is 104 s; the sandbox limit is 900 s. There are 0 breaches anywhere.
- Mirrored folds give the same PnL within 0–4.7%.
- The holdout was run once, at the end. Caveat: the full-tape shock diagnostic and one trace looked at shocks inside the holdout window. Nothing was tuned on them.

## How it got there (paired on the same folds)

| Step | Mean | Notes |
| --- | ---: | --- |
| Grid filter, s_mm-style quoting, takes at mean − 0.15 | +3.8k | 4 folds negative |
| Take threshold widened by the posterior sd (`take_sd` 1 → 2) | +16.4k | All folds positive |
| Wider quotes after detected jumps (`theta_turb`) | +18–24k | |
| Shorter excitation half-life (`exc_hl` 1.5) | +26.5k | Plateau over hl 1–3 and theta_turb 1.5–3 |
| Regime split: takes only in bursts, no passive quotes (`passive_t` 0) | +27.1k | |
| Impact c 0.037 → 0.03 | +28.1k | The likelihood fit peaks at 0.03 independently |
| Joint calm/volatile HMM filter with the teammates' fitted laws, plus the P(volatile) regime | +36.0k | With the regime flag off: +24.3k |
| Shock detection (drought → filter law, expected-jump window); sigma 0.3 | +36.7k | 27 of 27 shocks detected, about 0.38 s after onset |

Attribution with [fv_attrib.py](fv_attrib.py), version with the single-regime filter:
- Calm passive fills mark out +0.23/lot at 1 s, and calm takes +0.55–0.8.
- Turbulent passive fills mark out −2.4 to −3.0/lot.
- Turbulent takes mark out +1.1–1.35/lot and make most of the PnL in the turbulent folds.

## Rejected (each is a flag in PARAMS, default off)

- **Jump-count "stretch" regime** (`stretch_n`): ≈0.
- **Realised-volatility stretch regime** (`rv_thr` 1.5–3): −4 to −5k. Passive market making between jumps is still profitable; only the seconds after a jump are toxic.
- **Inventory skew** (`inv_skew` 0.3): −15k.
- **Unwind takes to free cap** (`unwind_u`): −0.1 to −1.5k in calm periods, −8k if also used in bursts.
- **Quantile take rule** (`take_q`): ±0. The posterior mean is the EV criterion.
- **Lower turbulent take threshold**: −1 to −3k.
- **Passive quotes in volatile periods** (`passive_t` 1, theta_t 1.5–2, reach_t 2.5): −2 to −4k.
- **Lower turbulent cap** (`cap_t` 50): −10k. The takes need the capacity.
- **IOC flatten on drought** (`shock_flat`): −1.5k.
- **Passive reduce-only quoting in droughts** (`shock_red` −0.5 to 0.3): −2.5k.
- **Filter variants:**
  - eta 0.0005–0.002: ≈0.
  - rho0 0.5–2: −2 to −7k, from false jumps in calm periods.
  - c 0.02–0.045: the peak is at 0.03, in v1.
  - lam 0.5–0.75 and sigma 0.3–0.8: flat.

## Known limits and ideas not yet tried

- Taker IOC limit prices are not visible live. The `taker_obs` likelihood is only usable offline in `r2_fv_fit.py`.
- Market cancels carry information that is not used. The cancel hazard rises with the adverse move of S.
- The posterior mixes the two regimes in one S grid (a factorised approximation). Two grids would be exact, at twice the cost.
- New tape: refit the laws, check `r2_fv_fit.py`'s likelihood and R², and keep the shock parameters only if droughts exist.
