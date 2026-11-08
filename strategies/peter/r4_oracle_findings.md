# Round 4: oracle headroom of r4_best, aspect by aspect (2026-11-08)

Base: `strategies/marek/round4/r4_best.py` (the best r4 file: 1,391.0k on the public r4 tape; r4_dev 1,371.5k,
r3_dev_best / r4_mom 1,365.9k). Each oracle replaces one decision input of r4_best with offline truth; the filter keeps
running. Whole r4 tape, PnL split with Marek's labels (calm / volatile / shock = onset −1 … +20 s).

Tools: `r4_oracle.py` (knobs), `r4_oracle_data.py` (truth npz from Marek's s_dynamics S path, rebuilt here: total
ll −1,071,314 vs his −1,071,071), `r4_oracle_eval.py` (full-tape runs, cv = calm + volatile per 500 s window, t-stat).

## Headroom (Δ vs r4_best, k)

| Aspect (oracle) | total | calm | vol | shock |
| --- | ---: | ---: | ---: | ---: |
| Next move S(t+h) − S(t), h 0.25 s | **+417** | +152 | +164 | +102 |
| … h 0.5 / 1 / 2 / 5 s | +330 / +112 / −182 / −464 | | | |
| … velocity part only, V_true · 0.25–0.5 s | **+67…+76** | +36…+40 | +34…+42 | ≈0 |
| Shock jump J known from the onset | **+91** | 0 | −1 | +93 |
| Level F := S(now), outside shocks (sd 0.15) | **+36…+53** | +32 | +18…+24 | ≈0 |
| Level, inside shocks too | −3 | | | −52 (the sniper is tuned to its own filter) |
| Offline model's causal filter as F | −60 (noshock −1) | +12 | +4 | |
| Regime := true volatile segments (decisions / + filter pv) | −20 / −14 | −1 / 0 | −1 / 0 | lottery |
| Shock onset := true onset | +0.1 | | | |

Reading: most of the "next move" is the future taker kicks (not predictable beyond the side law, whose lean `gm_kv` Marek
already found to lose). Regime and onset detection are already perfect for this strategy. Shock direction: no pre-shock
feature predicts J (28 shocks, |corr| ≤ 0.31, sign agreement 0.39–0.54). Calm level error S − F has OOS R² 0.07 on public
features (−0.065·(F − mid)).

## Improvement attempts, in headroom order (causal)

| Aspect | Change | cv Δ (t, windows won) | Kept |
| --- | --- | ---: | --- |
| Velocity, volatile | F += k · kv (filter's Kalman velocity), turbulent only: k 0.15 / 0.2 / 0.25 / 0.3 / 0.35 / 0.5 | +21 / +18 / +16 / +16 / +18 / +14 (t 1.4–2.0, 8–9/12) | **yes, k 0.2** |
| Velocity, both regimes | same, k 0.1 / 0.25 / 0.5 / 1.0 in calm too | calm −2…−5 | no |
| Velocity as a take veto only (`r4_veto`) | | −0.7 | no |
| Calm drift (15-lot agent tell) | online detector (15/10 lots at own touch 1 µs after a fill/cancel), calm F += side · 0.3 / 0.6 (2 s), 0.5 (4 s) | −0.2 / −4.0 / −6.9 | no |
| Calm level | shrink toward mid, `cm_k` / `cgk` −0.07, −0.15 | −1.8 / −2.1 / −7.5 | no |
| Regime, onset | nothing to gain | | — |
| Shock direction | nothing observable before the jump | | — |

Shock column of the kept variant: −11.2k, all from shock #6 (−16.9k, also moves on unrelated knobs); median per-shock Δ 0.

## Result: `strategies/peter/r4_kvl.py`

r4_best + `kvl_v 0.2`: **round 4 1,397.1k** (+6.1k; calm −2.8, volatile +20.1, shock −11.2 = #6; cv +17.3k, t 2.0, 9/12),
**round 3 845.6k** (r4_best 848.8k, cv −2.8k, t −0.7: r3 has no velocity).

## Follow-up on r4_volsig (Marek's file = the same lean, vs_kv 0.15; better on the hold part: +18.1k vs r4_kvl +2.9k)

All on the real r4 tape, `ex` = non-shock Δ vs r4_volsig (r4_full.py), t over 12 pieces. Fresh r4_synth tapes cannot
judge this: their generator has no velocity term (takers chase instead).

| Change | ex Δ (t) | |
| --- | ---: | --- |
| lean weight 0 / 0.25 | −22.6 (−3.5) / −8.9 | 0.15 is the optimum |
| kv filter: tau_v 1.5 / 3.0, sd_v 2.5 | −0.9 / −2.8 / −10.1 (−4.5) | totals +6…+19k were all shock luck |
| book lag `gm_kv` 0.05 / 0.1 / 0.15 next to the kv lean (offline it lifts the nowcast R² 0.15 → 0.22) | total +2.4 / −14.0 / 0 | dead, as on r4_best |
| take lean 0 / 0.3 / 0.5 (quotes 0.15) | −21.1 / −5.3 / −16.2 | most of the value is in the takes |
| quote lean 0 / 0.3 (takes 0.15) | −8.0 / −1.6 | |
| lean × kv² / (kv² + c·kPv), c 1 / 1 (k .25) / 3 (k .3) | −13.3 / −7.0 / −11.4 | |
| + F-momentum lean (vel_kv 0.03–0.1, tau 0.25–1 s) | −1.3 … −6.7 | |

Offline (`r4_vlog.py`, `r4_vnow.py`): the causal kv tracks the smoothed true velocity with corr 0.86 (R² 0.74; sd 1.9 vs
3.0 tick/s), lagging ~0.5 s; F momentum adds nothing (R² 0.738 vs 0.736). The linear velocity lean is at its optimum;
what is left of the +39k volatile velocity oracle is mostly that filter lag.
