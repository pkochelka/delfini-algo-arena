# Round 4 strategy: round-3 stack on round 4, hypotheses H1–H9

Written 2026-11-08. Tape `data/round4.csv` only (no synth), one regression run on `data/round3.csv`. Hypotheses from
`../hypothses.md`, generator facts from `../NOTES.md`.

**Bottom line.** Round 3's stack ports unchanged and earns *more* on round 4 (r3_hmm 1,371.5k vs 857.6k on r3), because
volatile carry is bigger. The momentum is already harvested: in volatile we sit at the cap 91 % of the time, on the
trend side, and position carry is 637k of the 775k volatile PnL. Every way of *leaning* on the momentum (velocity or
S − mid in the fair value, TWAP side, exits, quote pulls) loses. What helps is tracking S better: a Kalman velocity
state next to the S grid plus s_dynamics' round-4 volatile impact law.
**Best file: `strategies/marek/round4/r4_best.py`. Round 4 1,391.0k (+19.5k vs r3_hmm; calm +1.2k, volatile +23.5k,
shock −5.3k). Round 3 848.8k (−8.8k).**

## Tools

| Path | What |
| --- | --- |
| `eval4.py` | eval3 copy: round-4 tapes use `../flow/regimes_1s.csv` and `../flow/shock_events.csv`, round-3 tapes the round-3 files; adds `win_cv` (calm + volatile PnL per 500 s window) |
| `sweep.py` | many `--set` configs in one fastsim pool (eval3's sweep, repointed) |
| `tbl.py TAG..`, `summ.py TAG..` | deltas vs the last `base4` row (r3: `r3base`, `--r3`): calm / vol / cv = calm + vol / shock / shock without the biggest per-shock delta / median per-shock delta; `summ.py` adds window wins and t-stats (`tcv` = t of the 12 per-window calm + vol deltas) |
| `tools/mk_dev.py` | builds `strategies/marek/round4/r4_dev.py` from `r3_hmm.py` + all round-4 knob sets (all off = r3_hmm, verified identical: 1,371,508) |
| `tools/logf.py`, `ferr.py`, `ferr0.py`, `carry.py` | log our F / pos every 10 ms; filter error vs `s_dynamics/s_path.parquet`; predictability of S from our F; carry decomposition |
| `tools/counts.py`, `trace.py` | internal counters (droughts, TWAP chains, joiner drops); order/fill trace in a time window |
| `results.jsonl`, `cfg/*.jsonl` | every run (tags below) and the sweep configs |

Noise: shock PnL is a lottery. Shock #24 (5242.9 s, J = +116, volatile) swings ±22–25k on unrelated knobs, #6 and #10
by ±6–17k. Judge by **cv** (calm + volatile) and the median shock delta; window t-stats are in `summ.py`.

## Step 0. Baseline (tag `base`, `base4`)

| Strategy | Round 4 | calm | volatile | shock | breach | Round 3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `round3/r3_push.py` | 1,363.1k | 279.7k | 774.9k | 308.4k | 16.4k | 863.3k |
| `round3/r3_hmm.py` | **1,371.5k** | 279.2k | 775.4k | 316.8k | 16.6k | 857.6k |
| `round3/r3_dev_best.py` | 1,365.9k | 276.9k | 778.5k | 310.6k | 16.0k | 853.7k |

- Nothing is lost against round 3: calm is the same, volatile is ×1.8. r3_push's drought change (dr_theta / dr_margin)
  loses 8k of shock here, so the round-4 work starts from **r3_hmm**.
- Our F against s_path (v0, 10 ms grid): rmse calm 0.16, volatile 0.53. Inside TWAP windows calm 0.15 / vol 0.56,
  outside 0.16 / 0.53: no TWAP contamination (the filter uses only fill quantities for takers, and TWAP impact is real).
- All 28 droughts detected, no false alarm.
- Carry Σ pos·ΔS: volatile **+637k** (|pos| ≥ 95 for 91 % of volatile time), shock +161k, calm **−63k** (pinned 55 %).
  E[sign(pos)·ΔS] at +0.5 / 1 / 2 / 5 / 10 / 20 s in volatile: +2.1 / +3.6 / +5.7 / +6.8 / +4.5 / +0.8. The forward
  carry *rises* with the age of the position (1 s ahead: +2.3 at age < 1 s, +5.8 at 3–8 s), so holding positions do not
  give back; the 10–20 s give-back is in positions that flip anyway.
- Offline (our causal F, two halves): in volatile, S(t + 1 s) − F(t) ≈ 0.9·(F − EMA_0.5s F), out-of-sample R² 0.22–0.28;
  jointly with F − mid, S(t + 1 s) − F ≈ 0.3·(F − EMA_1s) + 1.4·(F − mid), R² 0.28. Calm R² ≤ 0.02.

## Hypotheses

Knob sets in `r4_dev.py` (all default off). Deltas vs r3_hmm on r4 in k (cv = calm + volatile).

| # | Test (knobs) | Result | calm | vol | cv | shock | Kept | Tags |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| H1 | 15-lot joiner: insert of 15 / 10 lots 1 µs after a fill or cancel row, at the same-side touch, dropped from the maker likelihood (`j15_on`; 694 drops vs ≈ 750 agent orders) | neutral | −0.8 | −0.7 | −1.5 | −22 (#24) | no | b2 |
| H1 | TWAP children out of the filter | not needed: filter error is the same inside TWAP windows (above) | | | | | no | — |
| H2 | F lean k·clip(F − mid) in volatile (`gm_kv` 0.2 / 0.5 / 1 / 2) | loses, monotone | −5..−17 | −10..−222 | −15..−239 | | no | h2a, b3 |
| H2 | F lean k·(F − EMA_0.5s F) in volatile (`vel_kv` 0.5 / 1 / 1.5; τ 1, k 0.2) | loses | −4..−10 | −14..−80 | −18..−90 (τ1: +3) | −22..−39 | no | h2a, b3 |
| H2 | calm leans (`gm_kc` 0.2, `vel_kc` 0.2) | loses | −14 / −19 | 0 | −14 / −19 | | no | h2a |
| H2 | lean only as a veto on takes against it (`r4_veto`) | no effect (≈ 0) | | | −1.5..0 | | no | b2 |
| H2 | filter drift from the EMA (`vd_k` 0.5 / 1) | loses | −1 | −4 / −22 | −5 / −24 | | no | h2a |
| H2 | **volatile prior diffusion** `lamj_v` 0.23 → 0.5 / 1 / 1.5 / 2 / 2.5 / 3 / 4 | **plateau 1.5–4** | −1..−3 | +2 / +8 / +13 / +16 / +16 / +16 / +17 | +13..+15 | lottery | superseded by kv | b3, b4 |
| H2 | **Kalman velocity state `kv_on`** (OU, τ 0.5 / 2.1 s, sd 0.3 / 3.0 tick/s calm / vol; S shifted by v·dt; v updated from each maker innovation × Cov(S,v)/Var S) | **wins**; plateau in sd 3–4, τ 2–3; gain 0.5 or sd 2 worse | +1 | +23..+26 with c_v | +24..+27 | −5 (#24) | **yes** | b8–b11, loo, final |
| H2 | volatile impact law of s_dynamics' fit with drift (`c_v` 0.05, `e_v` 0.64); full S v1 constants ≈ the same | wins with kv | +3 | +18 | +22 (alone with lamj 2: +7) | | **yes** | b5, b6, b10 |
| H3 | volatile aged IOC exit (`vx_u` 0, `vx_age` 3 / 5 / 8 s; `vx_u` 1 with momentum gate) | flat, as in r3 | 0 | −2 / +1 / 0 / −4 | −2.4 / +1.0 / +0.1 / −5.0 | −1..−25 | no | b2 |
| H4 | momentum takes | = the take side of the H2 leans: loses | | | | | no | h2a |
| H5 | volatile flatten quote off (`vf_on 0`) / gated harder (`vf_mom` 0, −1) / depth (`vf_th` 1.5, 2.5) | r3 setting stays best | −2.5 | −8 / −3 | −10.4 / −4.0 | | keep r3 | b2, b6 |
| H6 | online TWAP detector (IOC 500,000 ± 3 µs after a 160-lot fill, same side) + F lean `tw_k` 0.5 / 1 | loses in calm | −7 / −20 | 0 / −6 | −6.6 / −25.3 | | no | b2 |
| H6a | widen the side the TWAP hits during an episode (`tw_th` 0.5 / 1 / 5) | loses in calm | −5 / −12 / −11 | 0 | −5 / −15 / −12 | | no | b6 |
| H7 | IOC limit `ss_lim` 12, `xs_edge` 5, post-shock passive quote (`ps_on`, θ 1 or 2) | neutral / loses (ps: median shock −0.25k) | | | | −1 / 0 / −10..−15 | no | b2, b10 |
| H7 | per-shock: shock #8 (J = +137) loses 8–9k: short 100 into the drought, stale asks 9802 → next ask 9934, so only 12 lots to buy back. Drought detector has no false alarms, so the 148 s "no shock" window has nothing to cut | | | | | | — | trace |
| H8 | transient decay off (`tr_on 0`) | calm −3.8: decay still there (s_dynamics p 0.54, T 14 s) | −3.8 | +0.7 | −3.0 | | keep on | b2 |
| H9 | taker side law Φ((S − mid)/4.42) as a grid likelihood, TWAP children skipped (`ts_on`, slope 0.37 / 0.2) | neutral | −1.6 | −1.8 | −3.4 (on final stack −1.9) | | no | b2, b5, loo |
| — | retunes: take_t 0.15 / 0.5, take_sd_t 0.5 / 0.75 / 1.5, theta 0.2 / 0.35, eta 0.1 / 0.2, sigma 0.2–0.35, max_sd, pv_thr, q_cv/q_vc, jv_v, lq_on | flat or negative | | | | | no | b3–b7 |

Not tested: integer-second regime prior (r3 oracle worth ≤ 4k), the 15-lot joiner as a calm drift tell (≤ 31 episodes;
every calm lean lost), thicker deep book for exits (H3 found no exit value).

## Final stack: `strategies/marek/round4/r4_best.py`

`r4_dev.py` + `PARAMS.update({'c_v': 0.05, 'e_v': 0.64, 'kv_on': 1})`. Same imports as the round-3 files, no file
reads, no `modify_order`.

| Variant (tag `final`) | Round 4 | calm | volatile | shock | cv Δ | breach |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| r3_hmm (both off) | 1,371.5k | 279.2k | 775.4k | 316.8k | | 16.6k |
| − kv (c_v only) | 1,357.1k | 280.7k | 782.6k | 293.7k | +8.7k | |
| − c_v (kv only) | 1,360.9k | 276.8k | 787.8k | 296.2k | +10.0k | |
| **r4_best** | **1,391.0k** | **280.5k** | **798.9k** | **311.5k** | **+24.7k** | 12.0k |

- The two parts work together (+8.7 and +10.0 alone, +24.7 together). Median shock delta +0.2k; without the biggest
  shock delta the shock column is +19.7k. calm + volatile is up in 9 of 12 windows (t +2.9), total in 8 of 12.
- `lamj_v` 1.5 on top of kv adds nothing (cv +27.4 vs +24.7, shock median −0.15k), so it stays at 0.23.

**Round-3 regression** (tag `r3final`): 848.8k vs r3_hmm 857.6k (calm −1.8k, volatile −11.2k, shock +4.2k). kv alone
costs 3.0k cv on r3 (harmless: r3's S has no velocity); most of the cost is `c_v` 0.05, a round-4 fit (r3's own fit
was 0.062–0.074). If the next round looks like round 3 rather than 4, set `c_v` back to 0.0741, `e_v` 0.606 (r4 then
1,360.9k).

## Open

- Calm carry is −63k: passive fills against side-law takers. Every calm lean lost; a calm velocity is only ~0.2
  tick/s (s_dynamics), so this is probably the price of queue fills.
- Shock column: #24 (volatile, J = +116) is a ±25k coin flip on unrelated changes; #8 (J = +137) loses on a thin stale
  side after a full pre-shock position. Pre-shock inventory control was dead in r3; not retested.
- The TWAP detector splits episodes when a child does not fill (75 chains for 37 episodes); irrelevant while every
  TWAP use loses.
- A joint (S, v) grid would replace the Gaussian velocity side-car; the plateau in kv's sd / τ suggests little is left.
