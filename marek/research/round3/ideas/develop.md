# develop: building the best round-3 strategy from r3_best

Angle "develop", started 2026-11-08 03:00. Base `strategies/marek/round3/r3_best.py`: round 3 795.0k (calm 238.6k,
volatile 421.7k, shock 134.7k, breach 17.7k), synth v1 mean 923.1k (6 seeds), synth v2 mean 861.7k (4 held-out seeds),
round 2 761.0k. Work file `strategies/marek/round3/dev/dev.py`; every mechanism behind a PARAMS flag (off = r3_best).

Tools (this angle):
- `ideas/dev_run.sh P JOBFILE`: one eval3.py call per (config, tape), at most P simulator processes at a time.
  Job lines: `TAG FILE TAPESET [k=v ...]`, TAPESET = r3 | v1 | v2 | r2 | acc (r3 + v1) | all | comma list.
- `ideas/dev_summ.py TAGPREFIX...`: r3 / v1 mean / v2 mean / r2 with deltas vs r3_best per tape (and per part).

Acceptance: positive on round 3 AND on the synth v1 mean; finalists confirmed on v2 and round 2. Shock changes judged on
synth. Single-seed screens are dominated by the shock part (e.g. `calm_kp0` on seed 1: +15.9k total, of which the
calm part is +0.2k and the shock part +16.9k), so calm/volatile changes are read on their own part as well.

## Log

### 1. Adopted from bigmoves: C2 (shock sweep trigger)
`bm_sw=1 bm_sw_cr=1 bm_sw_big_only=1 ss_clip_e=2`, merged into dev.py (code from `ideas/bigmoves_trig.py`, flags
off = r3_best). bigmoves' numbers (tag `bigmoves_C2`): r3 829.0k (+34.0k, all shock), v1 mean +14.2k, v2 mean +1.8k,
r2 +7.1k. The earlier C1 (without `bm_sw_big_only`) lost the −107 shock on v2 seed 11 (−68k), C2 does not. Kept.

### 2. Filter constants from the final s_dynamics model (`s_dynamics/model_params.json`)
r3_best still runs the v3 calm constants and the round-2 regime switch rates.

| tag | change | r3 Δ (calm / vol / shock) | v1 mean Δ (n) (calm / vol / shock) |
| --- | --- | ---: | ---: |
| dev_f1 | calm c .0168, e .616, jv .0169 (ξ sd .130), tr_p .50, tr_T 16.7 | +7.9k (+3.3 / +2.6 / +2.0) | +5.4k (6) (+0.1 / +0.5 / +4.7) |
| dev_f2 | q_cv 0.0064, q_vc 0.0166 (dwell 156.7 / 60.3 s) | +3.6k (+1.3 / +0.9 / +1.3) | +3.8k (6) (+0.4 / +0.3 / +3.1) |
| dev_f4 | sigma 0.3 → 0.15 (extra diffusion; the generator has none) | +4.5k (+3.4 / +1.0 / +0.1) | see below |

All three pass the rule but are small and mostly shock part on synth. Combined test `dev_f124` below.

### 3. Adopted from volatile: momentum-gated flatten-only passive quote (candidate)
`vf_on=1 vf_th=2.0 vf_keep=1.5 vf_mom=-0.5 vf_pos=60`, ported into dev.py (from `ideas/volatile_tk.py`). volatile's
numbers (tag `volatile_vfm20`): r3 +12.1k (volatile part +9.1k), v1 mean +1.8k, v2 mean +2.8k; volatile part up on every
synth seed, shock part −4k on average (position entering shocks). Same idea as ceiling's `rp_mom`, so not stacked with
it. Candidate for the stack.

### 4. novel: nothing adopted
novel's only positive item is `ov_age 3` (reducing-side overflow quote in volatile, +3.2k r3 / +24k s1 mostly shock
noise). It competes with vfm20 for the same reducing side and room, so it is not stacked; vfm20 is the better-tested one.

### 5. Ported from calm (for stacking): `tr_take` (reduce-only calm IOC at a lower threshold)
Same code as `ideas/calm_a.py`. calm's combo `calm_c2` = `take=0.1 tr_take=0.0 tr_take_sd=1.0 theta_keep=0.0`:
r3 811.5k (+16.5k, calm +18.3k), s1 +7.3k (calm +10.1k). Waiting for calm's final pick.

### 6. Calm transient-impact fade (dead)
In volatile, fading Y is known to lose (volatile `vt_yk`). Here it is tested in calm only.

| tag | change | r3 Δ (calm) | v1 mean Δ (calm) |
| --- | --- | ---: | ---: |
| dev_cty05 | calm take reference F − 0.5·Y | −97.0k (−79.0k) | −94.5k (−81.9k) |
| dev_cty10 | F − 1.0·Y | −192.6k (−153.1k) | −190.9k (−163.5k) |
| dev_cqy03 | calm passive quotes around F − 0.3·Y | −99.0k (−88.8k) | −91.2k (−83.3k) |

Over the horizon that matters for calm fills (seconds), S drifts with the book lag, not back with the transient decay
(volatile agent: calm β 0.24 / 0.44 / 0.63 at 1 / 2 / 5 s). The symmetric take shift also loosens takes when Y has the
other sign; the stricter-only variant `dev_cty03s` is below.

### 7. Change of rules (user, ~03:45)
Synthetic tapes are dropped: evaluation on `data/round3.csv` only, 9 configs in parallel; acceptance = positive on
round 3. Shock-only gains are checked per shock (the 4127 s shock flips ±17k); calm/volatile gains are preferred. A final
round-2 regression run is still done. The synth numbers above stay as recorded.

Also adopted from calm (report done): `calm_c2` = `take=0.1 tr_take=0.0 tr_take_sd=1.0 theta_keep=0.0` (r3 +16.5k, calm
+18.3k; calm up on all 11 tapes the calm agent ran).

| tag | change (r3) | r3 | Δ | calm | vol | shock |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| dev_f124 | f1 + f2 + f4 | 802.3k | +7.3k | +4.9k | +1.7k | +0.6k |
| dev_cmk03 | calm passive centre F + 0.3·clip(F − mid, ±1) (ride calm momentum) | 764.9k | −30.1k | −37.3k | +1.0k | +6.2k |
| dev_cmk06 | same, 0.6 | 692.7k | −102.3k | −84.9k | 0 | −17.3k |
| **dev_S1** | **C2 + calm_c2** | **842.4k** | **+47.5k** | +17.4k | −0.9k | +31.0k |

The calm passive centre is sharply tuned at F: shifting it either way (by Y, by momentum) costs tens of k.

### 8. Round-3 batch (all vs r3_best 795.0k)

| tag | change | r3 | Δ | calm | vol | shock | verdict |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| dev_vfm20chk | vfm20 port check | 807.1k | +12.1k | +1.7k | +9.1k | +1.2k | = volatile_vfm20 (port exact) |
| dev_c2chk | calm_c2 port check | 811.5k | +16.5k | +18.3k | −0.6k | −1.2k | = calm_c2 (port exact) |
| dev_S2 | C2 + calm_c2 + vfm20 | 848.3k | +53.4k | +15.7k | +7.0k | +30.6k | |
| **dev_S3** | **S2 + f124 filter constants** | **853.7k** | **+58.7k** | +16.0k | +8.4k | +34.3k | **new base** |
| dev_cu05 | calm take unload 3–10 s after the take, IOC at ≥ F − 0.5 | 794.6k | −0.4k | −0.3k | 0 | −0.1k | dead (rarely fires: the take lots are recycled within 3 s anyway) |
| dev_cu03 | same, ≥ F − 0.3, 2–15 s | 794.7k | −0.3k | −0.4k | 0 | +0.1k | dead |
| dev_vx10 | volatile IOC exit once the position is ≥ 2 s old, at ≥ F − 1 | 781.7k | −13.2k | −2.6k | −7.5k | −3.2k | dead |
| dev_vx10m | same + momentum gate (F − mid)·sign(pos) ≤ 0 | 786.9k | −8.1k | −1.1k | −4.5k | −2.5k | dead |
| dev_vx05a3 | ≥ 3 s old, at ≥ F − 0.5 | 793.5k | −1.5k | −0.2k | −0.1k | −1.2k | dead (rarely fires) |
| dev_vx10a1 | ≥ 1 s old, F − 1, gated | 760.6k | −34.4k | −3.9k | −27.7k | −2.8k | dead |
| dev_cty03s | calm take F − 0.3·Y, stricter side only | 770.5k | −24.4k | −22.2k | +0.2k | −2.4k | dead |
| dev_cqym02 | calm passive centre F + 0.2·Y (lean with the kick) | 695.1k | −99.9k | −91.7k | −2.5k | −5.7k | dead (both signs of a Y shift lose ~90k) |
| dev_cgk01 | calm take reference + 0.1·clip(F − mid, ±1) | 800.9k | +5.9k | +6.7k | +0.1k | −0.9k | candidate |
| dev_cgk03 | same, 0.3 | 797.1k | +2.1k | +3.0k | +0.1k | −0.9k | |

The aged volatile exit (vx) confirms the volatile agent's finding from a different angle: even 2 s after a snipe, an
IOC exit at F − 1 costs more than the freed room earns. The passive, momentum-gated exit (vfm20) is the only cheap one.

### 9. One-at-a-time sweep around S3 (Δ vs dev_S3 853.7k; r3)

| tag | change | Δ | calm | vol | shock |
| --- | --- | ---: | ---: | ---: | ---: |
| d3_tsdt13 | take_sd_t 1.5 → 1.3 | +3.3k | +0.2k | +3.1k | 0 |
| d3_tsdt17 | take_sd_t 1.7 | −5.9k | −0.7k | −1.9k | −3.3k |
| d3_vfm03 | vf_mom −0.5 → −0.3 | +2.5k | −1.0k | +1.2k | +2.3k |
| d3_vfm10 | vf_mom −1.0 | −3.9k | −1.1k | −2.7k | −0.1k |
| d3_tkt25 | take_t 0.3 → 0.25 | +1.9k | +0.2k | +1.5k | +0.2k |
| d3_cgk05 / 10 / 20 | calm momentum take shift 0.05 / 0.1 / 0.2 | +1.7k / −0.3k / +0.3k | −0.5 / −1.0 / −0.6k | | |
| d3_tsd15 | calm take_sd 2 → 1.5 | +1.2k | −0.1k | +1.3k | 0 |
| d3_vfth18 / 22 | vf_th 1.8 / 2.2 (keep −0.2 below) | +1.2k / −1.3k | | | |
| d3_eta12 / 18 | eta 0.12 / 0.18 | −5.2k / +0.8k | | | |
| d3_vfp40 / 80 | vf_pos 40 / 80 | 0 / −4.0k | | | |
| d3_th20 | theta 0.25 → 0.20 | −2.4k | −1.9k | | |
| d3_isk05 | inv_skew 0.05 | −4.4k | −0.7k | | −3.3k |
| d3_trtm1 | tr_take −0.1 | −3.7k | −1.5k | | −3.1k |

Everything is within ±6k, i.e. S3 sits at a local optimum of these knobs. `cgk` (+5.9k alone on r3_best) gives nothing on
top of calm_c2's lower take threshold: both buy the same marginal calm takes.

### 10. More one-at-a-time around S3 (Δ vs dev_S3; r3). All flat or negative.

| tag | change | Δ | calm | vol | shock |
| --- | --- | ---: | ---: | ---: | ---: |
| d3_fr0 | fill_room 5 → 0 (no top-up of the target level) | −34.3k | −28.1k | +0.9k | −7.1k |
| d3_fr10 / fr20 | fill_room 10 / 20 | 0 / −5.9k | | | |
| d3_thsd05 | theta_sd 0.5 | −16.0k | −13.7k | | |
| d3_ps1 | post-shock passive quoting on | −6.2k | | | −6.1k |
| d3_msd04 / 09 | max_sd 0.4 / 0.9 | −2.8k / −0.6k | | | |
| d3_drm05, d3_drth06 | drought cut margin 0.5; drought theta 0.6 | −2.9k / −1.0k | | | |
| d3_rho02 | rho0 0.05 → 0.02 | −2.1k | +2.8k | | −4.2k |
| d3_bx01 / 04, d3_kd2 | burst_exc 0.1 / 0.4, keep_depth 2 | 0 (identical) | | | |
| d3_tk08 / 12 | calm take 0.08 / 0.12 | +1.5k / −4.8k | | | |
| d3_trsd07 / 13 | tr_take_sd 0.7 / 1.3 | +2.0k / +2.0k | | | |
| d3_pv06 | pv_thr 0.6 | −1.2k | | | |
| d3_tsdt11 / 12 / 13 / 15 / 17 | take_sd_t | −1.0 / −0.1 / +3.3 / 0 / −5.9k | | | |
| d3_vfm04 | vf_mom −0.4 | +1.3k | | | |
| d3_kpm05 / 10 / 20 | theta_keep −0.05 / −0.1 / −0.2 | −5.1k / −7.3k / −14.5k | | | |
| d3_thin0 / 10 / 40 | calm theta 0 / 0.1 / 0.4 when the market spread ≥ 2 (`th_in`) | −5.4k / −8.6k / −21.6k | | | |

take_sd_t 1.3 is an isolated +3.3k between −0.1k (1.2) and 0 (1.5): noise, not adopted.
The S3 shock gain (+34.3k) is shock 18 (4127 s, +28.1k) plus shocks 10 and 17 (+2.8k, +2.3k): the C2 sweep trigger,
which bigmoves validated on synth. Calm + volatile of S3: +24.4k.

### 11. Last checks (Δ vs dev_S3 853.7k; r3)

| tag | change | Δ | calm | vol | shock |
| --- | --- | ---: | ---: | ---: | ---: |
| d3_thr0 / 10 / 15 / −10 | calm reducing-side theta when \|pos\| ≥ 90 (`th_red`) | −18.9k / −22.7k / −18.0k / −25.4k | −1.7 to −6.3k | ~0 | −15.6 to −20.2k |
| d3_thr0p60 / 10p60 | same from \|pos\| ≥ 60 | −23.2k / −22.7k | | | |
| d3_isk15 / 25 | inv_skew 0.15 / 0.25 | −16.1k / −38.0k | −10.6k / −17.0k | | |
| d3_vsmall | take_sd_t 1.3 + vf_mom −0.3 | +2.4k | −0.3k | +1.2k | +1.5k |
| d3_small | + take 0.08 + tr_take_sd 0.7 | +1.1k | 0 | +1.5k | −0.4k |

The small positives do not stack beyond noise; not adopted.

**Leave-one-out of S3** (Δ when the component is removed):

| removed | r3 | Δ | calm | vol | shock |
| --- | ---: | ---: | ---: | ---: | ---: |
| bigmoves C2 | 822.1k | −31.6k | +0.2k | −0.2k | −31.7k |
| calm c2 | 843.7k | −10.0k | −12.9k | +1.4k | +1.5k |
| volatile vfm20 | 845.3k | −8.5k | +0.5k | −7.0k | −1.9k |
| filter constants (= dev_S2) | 848.3k | −5.4k | −0.2k | −1.5k | −3.7k |

**Robustness of S3 vs r3_best:** ahead in 12 of 12 500 s windows on round 3 (+0.5k … +4.8k each; +29.8k in the window
with the 4127 s shock), and in 8 of 12 on round 2 (window deltas −2.1k … +9.8k).

## Summary

**Final file:** `strategies/marek/round3/r3_dev_best.py` = r3_best + four kept changes, each behind PARAMS flags (all
flags off = r3_best); the settings are the last `PARAMS.update` block of the file. Reference `exchange-simulator`
run on round 3: 853,704.66 (fastsim/eval3: 853,705, identical). 75.6 kB, same imports as r3_best, no `modify_order`.

| tape | r3_best | **r3_dev_best** | Δ |
| --- | ---: | ---: | ---: |
| round 3 total (net of breach) | 795.0k | **853.7k** | **+58.7k (+7.4 %)** |
| — calm | 238.6k | 254.5k | +16.0k |
| — volatile | 421.7k | 430.1k | +8.4k |
| — shock windows | 134.7k | 169.0k | +34.3k (shock 18 +28.1k) |
| — breach penalty | 17.7k | 17.1k | |
| round 2 (regression) | 761.0k | 778.4k | +17.4k (calm +4.6k, vol +4.4k, shock +8.3k) |
| synth v1 / v2, full stack | 923.1k / 861.7k | not run (user stopped synth runs) | per component below |

| kept change | flags | r3 Δ alone | Δ in the stack (leave-one-out) | synth evidence (by its owner) |
| --- | --- | ---: | ---: | --- |
| bigmoves C2: shock sweep trigger + clip edge credit | `bm_sw 1, bm_sw_cr 1, bm_sw_big_only 1, ss_clip_e 2` | +34.0k (shock) | +31.6k | v1 +14.2k, v2 +1.8k, r2 +7.1k |
| calm c2: take 0.10, reduce-only calm IOC at edge ≥ 0 + 1·sd, theta_keep 0 | `take, tr_take 0, tr_take_sd 1, theta_keep 0` | +16.5k (calm +18.3k) | +10.0k | calm up on 11/11 tapes; v1 +6.1k, v2 +14.5k |
| volatile vfm20: momentum-gated flatten-only passive quote | `vf_on 1, vf_th 2.0, vf_keep 1.5, vf_mom −0.5, vf_pos 60` | +12.1k (vol +9.1k) | +8.5k | vol part up on every synth seed; v1 +1.8k, v2 +2.8k |
| filter constants of the final s_dynamics model | `c .0168, e .616, jv .0169, tr_p .50, tr_T 16.7, q_cv .0064, q_vc .0166, sigma .15` | +7.3k | +5.4k | v1 +4.6k (6 seeds, f124) |

**What did not work (this angle; numbers in the log above):**
- Fading the transient impact Y in calm, in takes or quotes, either sign: −24k to −193k. The calm passive centre is
  sharply tuned at F; any shift (Y, book-lag momentum `cm_k`) costs 30–100k.
- Calm take-inventory unload 2–15 s after a take (`cu_*`): ±0.4k (the take lots are already recycled within seconds).
- Volatile IOC exit once the position is 1–3 s old (`vx_*`), with or without a momentum gate: −1.5k to −34k. Even 2 s
  after a snipe an IOC exit costs more than the freed room earns.
- Calm momentum in the take reference (`cgk`): +5.9k alone, 0 on top of calm c2.
- Inside-spread theta (`th_in`), negative theta_keep, pinned reducing theta (`th_red`), stronger inventory skew,
  fill_room changes, theta_sd, post-shock quoting, max_sd, drought cut/theta, rho0, eta, take_t / take_sd_t, vf_*
  neighbours: all within noise or negative. S3 is at a local optimum of every existing knob.

**Where the gap to 2× is (agreed by all five angles):** volatile room. r3_best/r3_dev_best are pinned at ±100 for 90 %
of volatile time; each extra 100 lots of held room is worth ≈ 425k gross, but every exit tried (IOC at any age,
unconditional passive, oversize into the ladder) costs more than the room is worth under the 10k/s penalty. Calm's
ceiling is queue priority (≈ 365k vs ≈ 123k realised), and no rule that spends room to buy priority has paid.

**Remaining leads (untested or weak):**
1. Shocks: ≈ 45k left on r3 (bigmoves §2.2: shocks 14 and 2 are latency-bound on invisible inserts; 20 at 4559 s still
   leaves 4k). Judge on synth, since r3 has 27 shocks.
2. Volatile: a passive exit that is cheaper than vfm20's F ± 2 — e.g. the exit quote placed exactly at the takers' limit
   round(F ± 2.55) and only after the momentum gate (novel's `ov_age` and volatile's `vf` combined); both alone are
   +2–12k and compete for the same room, so test one combined rule rather than stacking.
3. Calm: queue priority without spending room (the only large calm pool); every variant tried so far spent room.
4. Synth confirmation of the full stack (v1 + v2) if synth runs are allowed again: each component was validated there,
   the combination was not.
